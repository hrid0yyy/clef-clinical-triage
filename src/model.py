import os
import sys
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

def load_clef_model_and_processor(model_id="Cloudflare/clef-flash", cache_dir="./model_cache", device="cuda", config=None):
    """
    Downloads Cloudflare/clef-flash repository and loads release model + processor.
    Supports 4-bit (QLoRA), 8-bit, and native bfloat16 precision.
    """
    print(f"Downloading/Locating model repository: {model_id}...")
    repo_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.msgpack", "*.h5"]
    )
    print(f"Model files located at: {repo_path}")
    
    # Add repo path to sys.path to access joint_schema_model.py
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        
    try:
        from joint_schema_model import load_release_model, encode_record, collate_records
    except ImportError as e:
        raise ImportError(f"Failed to import joint_schema_model from {repo_path}: {e}")

    # Build kwargs for quantization if requested
    from_pretrained_kwargs = {}
    quantization = "4bit"
    if config and "training" in config:
        quantization = config["training"].get("quantization", "4bit").lower()

    if torch.cuda.is_available() and quantization in ["4bit", "8bit"]:
        try:
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                print("Enabling 4-bit NormalFloat (NF4) quantization for QLoRA...")
                from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            elif quantization == "8bit":
                print("Enabling 8-bit quantization...")
                from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"Quantization notice: {e}. Falling back to native bfloat16.")

    target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading weights into memory on device={device} (dtype={target_dtype})...")
    
    try:
        model, processor = load_release_model(
            repo_path,
            device=device,
            dtype=target_dtype,
            **from_pretrained_kwargs
        )
    except TypeError:
        model, processor = load_release_model(repo_path, device=device)

    # Patch forward pass to guarantee precise dtype compatibility across QLoRA layers and Schema Head
    def robust_clef_forward(self, batch):
        base_model = (
            self.language_model.get_base_model()
            if hasattr(self.language_model, "get_base_model")
            else self.language_model
        )
        media = batch.get("media") or {}
        text_model = base_model.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        outputs = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        target_dtype = self.head.hidden_norm.weight.dtype if hasattr(self.head, "hidden_norm") else torch.bfloat16
        hidden_states = outputs.last_hidden_state.to(dtype=target_dtype)
        emb_weight = base_model.get_output_embeddings().weight.to(dtype=target_dtype)
        return self.head(
            hidden_states,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            emb_weight,
        )

    model.__class__.forward = robust_clef_forward

    print("Clef-Flash base model and processor loaded successfully.")
    return model, processor, encode_record, collate_records, repo_path

def setup_trainable_model(model, config):
    """
    Configures training mode (LoRA PEFT, Head-Only, or Full Fine-Tuning).
    """
    mode = config["training"].get("mode", "lora").lower()
    print(f"\nConfiguring training mode: [{mode.upper()}]")
    
    # Locate backbone and head modules
    # In ClefModel: self.language_model is the backbone, self.head is the schema head
    backbone = getattr(model, "language_model", getattr(model, "backbone", getattr(model, "model", None)))
    head = getattr(model, "head", getattr(model, "schema_head", None))
    
    if mode == "head_only":
        print("Freezing entire backbone. Only JointSchemaHead parameters will be trained.")
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = "head" in name.lower()

    elif mode == "lora":
        print("Attaching LoRA adapters to backbone attention projections.")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare k-bit model if quantized
            quantization = config["training"].get("quantization", "none")
            if quantization in ["4bit", "8bit"] and backbone is not None:
                try:
                    backbone = prepare_model_for_kbit_training(backbone)
                except Exception as e:
                    print(f"Notice: prepare_model_for_kbit_training: {e}")

            lora_cfg = config["training"]
            peft_config = LoraConfig(
                r=lora_cfg.get("lora_r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            # Wrap backbone with LoRA
            if hasattr(model, "language_model"):
                model.language_model = get_peft_model(backbone, peft_config)
            elif hasattr(model, "backbone"):
                model.backbone = get_peft_model(backbone, peft_config)
            else:
                model = get_peft_model(model, peft_config)
                
            # Ensure the JointSchemaHead remains fully trainable
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True
            else:
                for name, param in model.named_parameters():
                    if "head" in name.lower():
                        param.requires_grad = True
                        
        except ImportError:
            print("WARNING: 'peft' package not found. Falling back to Head-Only training.")
            for name, param in model.named_parameters():
                param.requires_grad = "head" in name.lower()

    elif mode == "full":
        print("Full fine-tuning: Unfreezing all 9B parameters (Requires high VRAM GPU).")
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    # Report parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)\n")
    
    return model
