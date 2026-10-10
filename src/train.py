import os
import sys
import json
import yaml
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import SymptomDataset, create_collate_fn
from model import load_clef_model_and_processor, setup_trainable_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_logits(out):
    """
    Robustly extracts and stacks question logits across different Clef batch return formats.
    """
    if isinstance(out, (list, tuple)):
        # List of outputs per sample in batch
        batch_logits = []
        for item in out:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                # First question logits
                q_log = item[0]
            elif isinstance(item, torch.Tensor):
                q_log = item
            else:
                q_log = item
            batch_logits.append(q_log)
        if len(batch_logits) > 0 and isinstance(batch_logits[0], torch.Tensor):
            if batch_logits[0].ndim == 1:
                return torch.stack(batch_logits, dim=0)
            elif batch_logits[0].ndim == 2:
                return torch.cat(batch_logits, dim=0)
        return out[0]
    elif isinstance(out, torch.Tensor):
        return out
    return out

def evaluate(model, test_loader, device):
    """Evaluates model performance on the test split."""
    model.eval()
    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"]
            targets = batch["targets"].to(device)
            
            # Forward pass through Clef decision model
            out = model(inputs)
            logits = extract_logits(out)
            
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            
            # Top-1 predictions
            preds = torch.argmax(logits, dim=-1)
            top1_correct += (preds == targets).sum().item()
            
            # Top-3 predictions
            _, top3_indices = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)
            for i in range(targets.size(0)):
                if targets[i] in top3_indices[i]:
                    top3_correct += 1
                    
            total_samples += targets.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
    avg_loss = total_loss / max(total_samples, 1)
    top1_acc = (top1_correct / total_samples) * 100.0
    top3_acc = (top3_correct / total_samples) * 100.0
    
    return {
        "eval_loss": avg_loss,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "total_samples": total_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Clef-Flash on Symptom2Disease")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"].get("seed", 42))
    device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Load diagnostic criteria dictionary
    criteria_path = config["data"]["criteria_path"]
    with open(criteria_path, "r", encoding="utf-8") as f:
        criteria_dict = json.load(f)
    print(f"Loaded {len(criteria_dict)} verified disease criteria.")

    # Load model and components
    model, processor, encode_record_fn, collate_records_fn, repo_path = load_clef_model_and_processor(
        model_id=config["model"]["model_id"],
        cache_dir=config["model"]["cache_dir"],
        device=device,
        config=config
    )

    # Setup trainable parameters (LoRA, Head-only, or Full)
    model = setup_trainable_model(model, config)

    # Initialize Datasets
    use_criteria = config["data"].get("use_criteria", True)
    instruction = config["data"].get("question_instruction", "Which disease does this patient have?")
    
    train_dataset = SymptomDataset(
        csv_path=config["data"]["train_path"],
        criteria_dict=criteria_dict,
        question_instruction=instruction,
        encode_record_fn=encode_record_fn,
        processor=processor,
        use_criteria=use_criteria
    )
    test_dataset = SymptomDataset(
        csv_path=config["data"]["test_path"],
        criteria_dict=criteria_dict,
        question_instruction=instruction,
        encode_record_fn=encode_record_fn,
        processor=processor,
        use_criteria=use_criteria
    )

    # DataLoaders
    pad_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0
    collate_fn = create_collate_fn(collate_records_fn, pad_id, device)
    
    batch_size = config["training"].get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initial zero-shot evaluation baseline before any training
    print("\n--- Evaluating Zero-Shot Pre-Trained Baseline on Test Set ---")
    baseline_metrics = evaluate(model, test_loader, device)
    print(f"Baseline Top-1 Accuracy: {baseline_metrics['top1_accuracy']:.2f}% | Top-3 Accuracy: {baseline_metrics['top3_accuracy']:.2f}%\n")

    # Optimizer and Learning Rate Setup
    lr = float(config["training"].get("learning_rate", 1e-4))
    head_lr = float(config["training"].get("head_learning_rate", 3e-4))
    weight_decay = float(config["training"].get("weight_decay", 0.01))

    # Parameter groupings for differential learning rates
    head_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name.lower():
            head_params.append(p)
        else:
            other_params.append(p)

    optimizer_grouped_params = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params)

    num_epochs = config["training"].get("num_epochs", 5)
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 4)
    criterion = nn.CrossEntropyLoss()

    # Cosine learning rate scheduler with warmup
    total_training_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    warmup_steps = int(total_training_steps * config["training"].get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    output_dir = config["output"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    best_acc = baseline_metrics["top1_accuracy"]
    history = []

    print(f"Beginning fine-tuning for {num_epochs} epochs (Total steps: {total_training_steps}, Warmup: {warmup_steps})...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for step, batch in enumerate(pbar):
            inputs = batch["inputs"]
            targets = batch["targets"].to(device)

            out = model(inputs)
            logits = extract_logits(out)

            loss = criterion(logits, targets) / grad_accum_steps
            loss.backward()
            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"train_loss": f"{running_loss / (step + 1):.4f}"})

        # Epoch Evaluation
        eval_res = evaluate(model, test_loader, device)
        epoch_time = time.time() - start_time
        print(f"\n[Epoch {epoch}] Test Loss: {eval_res['eval_loss']:.4f} | Top-1 Acc: {eval_res['top1_accuracy']:.2f}% | Top-3 Acc: {eval_res['top3_accuracy']:.2f}%")

        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "eval_loss": eval_res["eval_loss"],
            "top1_accuracy": eval_res["top1_accuracy"],
            "top3_accuracy": eval_res["top3_accuracy"],
            "elapsed_time_sec": epoch_time
        }
        history.append(record)

        # Checkpointing
        if eval_res["top1_accuracy"] > best_acc:
            best_acc = eval_res["top1_accuracy"]
            best_path = os.path.join(output_dir, "best_clef_flash_model.pt")
            trainable_names = {k for k, v in model.named_parameters() if v.requires_grad}
            trainable_state_dict = {
                k: v.cpu() for k, v in model.state_dict().items()
                if k in trainable_names or "lora" in k.lower() or "head" in k.lower()
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainable_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_top1_acc": best_acc,
                "config": config
            }, best_path)
            print(f"★ New best accuracy achieved ({best_acc:.2f}%)! Saved to {best_path}\n")

    # Save final model & training log
    final_path = os.path.join(output_dir, "final_clef_flash_model.pt")
    trainable_names = {k for k, v in model.named_parameters() if v.requires_grad}
    final_state_dict = {
        k: v.cpu() for k, v in model.state_dict().items()
        if k in trainable_names or "lora" in k.lower() or "head" in k.lower()
    }
    torch.save(final_state_dict, final_path)
    
    log_path = os.path.join(output_dir, "training_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_metrics": baseline_metrics,
            "best_top1_accuracy": best_acc,
            "history": history
        }, f, indent=2)

    print("==================================================")
    print(f"Fine-tuning complete! Best Test Top-1 Accuracy: {best_acc:.2f}%")
    print(f"Artifacts saved in: {output_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
