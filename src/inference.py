import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import load_clef_model_and_processor
from dataset import safe_encode_record
from train import extract_logits

def run_inference(patient_text, checkpoint_path=None, config_path="configs/training_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Load criteria
    with open(config["data"]["criteria_path"], "r", encoding="utf-8") as f:
        criteria_dict = json.load(f)
    classes = sorted(list(criteria_dict.keys()))

    # Load base model & components
    model, processor, encode_record_fn, collate_records_fn, _ = load_clef_model_and_processor(
        model_id=config["model"]["model_id"],
        cache_dir=config["model"]["cache_dir"],
        device=device,
        config=config
    )

    # Setup trainable parameters if needed so LoRA/head exist for state_dict matching
    from model import setup_trainable_model
    model = setup_trainable_model(model, config)

    # Load fine-tuned weights if provided
    if checkpoint_path and not os.path.exists(checkpoint_path):
        alt_names = [
            os.path.join(os.path.dirname(checkpoint_path), "best_clef_flash_lora_and_head.pt"),
            os.path.join(os.path.dirname(checkpoint_path), "best_clef_flash_model.pt"),
            "checkpoints/best_clef_flash_lora_and_head.pt",
            "checkpoints/best_clef_flash_model.pt"
        ]
        for alt in alt_names:
            if os.path.exists(alt):
                checkpoint_path = alt
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Running in zero-shot base mode.")

    model.eval()

    # Formulate record (respecting use_criteria configuration)
    use_criteria = config["data"].get("use_criteria", True)
    criteria_map = {c: criteria_dict[c] for c in classes} if use_criteria else {c: None for c in classes}

    record = {
        "state": patient_text,
        "questions": {
            "disease": {
                "type": "choice",
                "instructions": config["data"].get("question_instruction", "Which disease does this patient have?"),
                "criteria": criteria_map
            }
        }
    }

    encoded = safe_encode_record(encode_record_fn, record, processor)
    pad_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0
    batch_inputs = collate_records_fn([encoded], pad_id, device)

    with torch.no_grad():
        out = model(batch_inputs)
        logits = extract_logits(out)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=3)
    
    top_pred = classes[top_indices[0].item()]
    top_prob = top_probs[0].item()

    print("\n================ CLINICAL PREDICTION ================")
    print(f"Patient Text: \"{patient_text}\"")
    print(f"Top Diagnosis: {top_pred} ({top_prob * 100:.2f}% probability)")
    print("\nDifferential Diagnosis (Top-3):")
    for rank, (idx, p) in enumerate(zip(top_indices, top_probs), 1):
        print(f"  {rank}. {classes[idx.item()]}: {p.item() * 100:.2f}%")
    print("=====================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Patient symptom narrative")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_clef_flash_model.pt")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()

    run_inference(args.text, args.checkpoint, args.config)
