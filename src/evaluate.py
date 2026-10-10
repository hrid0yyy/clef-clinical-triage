import os
import sys
import json
import yaml
import argparse
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import SymptomDataset, create_collate_fn
from model import load_clef_model_and_processor
from train import extract_logits

def evaluate_checkpoint(checkpoint_path, config_path="configs/training_config.yaml", output_json="evaluation_results.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoint: {checkpoint_path} on {device}")

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

    # Setup trainable parameters (ensuring LoRA adapters/head exist for state_dict matching)
    from model import setup_trainable_model
    model = setup_trainable_model(model, config)

    # Load fine-tuned weights
    if not os.path.exists(checkpoint_path):
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

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Fine-tuned weights loaded successfully from {checkpoint_path}.")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running zero-shot evaluation.")

    model.eval()

    # Load test dataset (80/20 test split)
    test_dataset = SymptomDataset(
        csv_path=config["data"]["test_path"],
        criteria_dict=criteria_dict,
        question_instruction=config["data"].get("question_instruction", ""),
        encode_record_fn=encode_record_fn,
        processor=processor,
        use_criteria=config["data"].get("use_criteria", True)
    )

    pad_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0
    collate_fn = create_collate_fn(collate_records_fn, pad_id, device)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_targets = []
    top3_correct = 0

    print("Running inference across 240 test cases...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"]
            targets = batch["targets"].to(device)

            out = model(inputs)
            logits = extract_logits(out)

            preds = torch.argmax(logits, dim=-1)
            _, top3_indices = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)

            for i in range(targets.size(0)):
                if targets[i] in top3_indices[i]:
                    top3_correct += 1

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    total = len(all_targets)
    top1_correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    top1_acc = (top1_correct / total) * 100.0
    top3_acc = (top3_correct / total) * 100.0

    # Detailed per-class metrics
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_targets, all_preds).tolist()

    results = {
        "checkpoint": checkpoint_path,
        "total_test_samples": total,
        "top1_accuracy": round(top1_acc, 2),
        "top3_accuracy": round(top3_acc, 2),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {cls: report[cls] for cls in classes if cls in report},
        "confusion_matrix": conf_mat
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n========================================================")
    print(f"EVALUATION RESULTS FOR {checkpoint_path}")
    print("========================================================")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"Top-3 Accuracy: {top3_acc:.2f}% ({top3_correct}/{total})")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Saved complete report to {output_json}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_clef_flash_model.pt")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.config, args.output)
