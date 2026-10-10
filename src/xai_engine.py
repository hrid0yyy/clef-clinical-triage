import os
import sys
import json
import yaml
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import load_clef_model_and_processor, setup_trainable_model
from dataset import safe_encode_record
from train import extract_logits

def load_environment(checkpoint_path="checkpoints/best_clef_flash_lora_and_head.pt", config_path="configs/training_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Loading base model on {device}...")

    # Load criteria
    with open(config["data"]["criteria_path"], "r", encoding="utf-8") as f:
        criteria_dict = json.load(f)
    classes = sorted(list(criteria_dict.keys()))

    # Load model and processor
    model, processor, encode_record_fn, collate_records_fn, _ = load_clef_model_and_processor(
        model_id=config["model"]["model_id"],
        cache_dir=config["model"]["cache_dir"],
        device=device,
        config=config
    )

    model = setup_trainable_model(model, config)

    if os.path.exists(checkpoint_path):
        print(f"[XAI] Loading fine-tuned weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("[XAI] Checkpoint loaded successfully!")
    else:
        print(f"[XAI WARNING] Checkpoint {checkpoint_path} not found!")

    model.eval()
    return model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device

def predict_probabilities(model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, text):
    use_criteria = config["data"].get("use_criteria", False)
    criteria_map = {c: criteria_dict[c] for c in classes} if use_criteria else {c: None for c in classes}
    record = {
        "state": text,
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
    batch = collate_records_fn([encoded], pad_id, device)

    with torch.no_grad():
        out = model(batch)
        logits = extract_logits(out)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return probs, batch

def run_perturbation_xai(model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases):
    print("\n" + "="*70)
    print("  EXECUTING METHOD 1: PERTURBATION-BASED FEATURE ATTRIBUTION (LOSO)")
    print("="*70)

    results = []

    for case in test_cases:
        patient_text = case["text"]
        true_label = case.get("label", "Unknown")
        print(f"\n--- Analyzing Case: True Label: [{true_label}] ---")
        print(f"Full Text: \"{patient_text}\"")

        # 1. Baseline prediction
        base_probs, _ = predict_probabilities(
            model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, patient_text
        )
        top1_idx = int(np.argmax(base_probs))
        top1_pred = classes[top1_idx]
        top1_prob = float(base_probs[top1_idx])

        # Top-3 baseline
        top3_indices = np.argsort(base_probs)[::-1][:3]
        top3_baseline = [{"disease": classes[i], "probability": float(base_probs[i])} for i in top3_indices]

        print(f"Baseline Prediction: {top1_pred} ({top1_prob*100:.2f}%)")

        # 2. Extract clinical symptom clauses
        # Split on commas, periods, semicolons, and common conjunctions
        clauses = [c.strip() for c in re.split(r"[,.;]\s*", patient_text) if len(c.strip()) > 3]
        if not clauses:
            clauses = [patient_text]

        clause_attributions = []

        for clause in clauses:
            # Create perturbed state by removing this specific clause
            ablated_text = patient_text.replace(clause, "").replace("  ", " ").strip()
            # If removing emptied it or left meaningless punctuation, clean up
            if not ablated_text:
                continue

            ablated_probs, _ = predict_probabilities(
                model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, ablated_text
            )
            ablated_top1_idx = int(np.argmax(ablated_probs))
            ablated_top1_pred = classes[ablated_top1_idx]
            ablated_target_prob = float(ablated_probs[top1_idx])
            
            # Marginal attribution score (drop in target disease confidence)
            marginal_drop = top1_prob - ablated_target_prob
            pct_drop = (marginal_drop / max(top1_prob, 1e-6)) * 100.0

            # Alternative disease surge
            alt_indices = np.argsort(ablated_probs)[::-1][:2]
            alt_surges = [{"disease": classes[i], "probability": float(ablated_probs[i])} for i in alt_indices if classes[i] != top1_pred]

            clause_attributions.append({
                "clause": clause,
                "ablated_target_prob": ablated_target_prob,
                "marginal_drop": marginal_drop,
                "percent_confidence_drop": pct_drop,
                "prediction_flipped": (ablated_top1_pred != top1_pred),
                "new_top_prediction": ablated_top1_pred,
                "new_top_prob": float(ablated_probs[ablated_top1_idx]),
                "surging_alternatives": alt_surges
            })

            flip_str = f" [FLIPPED to {ablated_top1_pred} ({ablated_probs[ablated_top1_idx]*100:.1f}%)]" if ablated_top1_pred != top1_pred else ""
            print(f"  • Ablated: \"{clause}\" -> Target Prob: {ablated_target_prob*100:.1f}% (Drop: {marginal_drop*100:+.2f}% / {pct_drop:.1f}% drop){flip_str}")

        # Sort clauses by marginal drop (highest importance first)
        clause_attributions.sort(key=lambda x: x["marginal_drop"], reverse=True)

        results.append({
            "case_id": case.get("id", len(results) + 1),
            "true_label": true_label,
            "patient_text": patient_text,
            "baseline_prediction": top1_pred,
            "baseline_probability": top1_prob,
            "top3_baseline": top3_baseline,
            "attributions": clause_attributions
        })

    return results

def run_whitebox_gradient_xai(model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases):
    print("\n" + "="*70)
    print("  EXECUTING METHOD 2: WHITE-BOX INPUT-GRADIENT ATTRIBUTION & SALIENCY")
    print("="*70)

    # Locate base model embedding layer
    base_model = (
        model.language_model.get_base_model()
        if hasattr(model.language_model, "get_base_model")
        else model.language_model
    )
    text_model = base_model.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    embed_tokens = text_model.embed_tokens

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    results = []

    for case in test_cases:
        patient_text = case["text"]
        true_label = case.get("label", "Unknown")
        print(f"\n--- Gradient Analysis for: [{true_label}] ---")

        # Encode record
        use_criteria = config["data"].get("use_criteria", False)
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
        pad_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0
        batch = collate_records_fn([encoded], pad_id, device)

        input_ids = batch["input_ids"][0] # (seq_len,)
        tokens = [tokenizer.decode([tid]) for tid in input_ids]

        # Hook embeddings
        embeddings_holder = []
        def hook_fn(module, input, output):
            output.retain_grad()
            embeddings_holder.append(output)

        handle = embed_tokens.register_forward_hook(hook_fn)

        model.zero_grad()
        out = model(batch)
        logits = extract_logits(out) # (1, num_classes)
        target_idx = torch.argmax(logits, dim=-1).item()
        target_disease = classes[target_idx]
        target_logit = logits[0, target_idx]

        # Backward pass to obtain gradient on input embeddings
        target_logit.backward()
        handle.remove()

        embedded = embeddings_holder[0] # (1, seq_len, hidden_dim)
        grad = embedded.grad # (1, seq_len, hidden_dim)

        if grad is not None:
            # Input * Gradient attribution score per token: sum_d (e_d * g_d)
            attr_scores = (embedded * grad).sum(dim=-1).squeeze(0).detach().cpu().numpy()
            grad_norm = torch.norm(grad, dim=-1).squeeze(0).detach().cpu().numpy()
        else:
            print("[XAI WARNING] Embedding grad was None, falling back to uniform attribution.")
            attr_scores = np.ones(len(input_ids))
            grad_norm = np.ones(len(input_ids))

        # Filter only tokens belonging to the patient symptom text (state)
        # Identify state token boundaries in sequence
        # We find tokens by searching for matching text pieces
        token_attributions = []
        for i, (tok, tid, score, gnorm) in enumerate(zip(tokens, input_ids.tolist(), attr_scores.tolist(), grad_norm.tolist())):
            token_attributions.append({
                "token_id": tid,
                "token": tok,
                "input_x_grad": float(score),
                "grad_norm": float(gnorm)
            })

        # Calculate top salient tokens
        # Filter out special control tokens and whitespace
        salient_tokens = [
            t for t in token_attributions 
            if not t["token"].strip().startswith("<|") and len(t["token"].strip()) > 1
        ]
        salient_tokens.sort(key=lambda x: abs(x["input_x_grad"]), reverse=True)

        print(f"Top-5 Salient Tokens for Predicted Disease [{target_disease}]:")
        for st in salient_tokens[:5]:
            print(f"   Token: '{st['token'].strip()}' | Score: {st['input_x_grad']:.4f} | GradNorm: {st['grad_norm']:.4f}")

        results.append({
            "case_id": case.get("id", len(results) + 1),
            "true_label": true_label,
            "patient_text": patient_text,
            "predicted_disease": target_disease,
            "target_logit": float(target_logit.detach().cpu().item()),
            "token_attributions": token_attributions,
            "top_salient_tokens": salient_tokens[:10]
        })

    return results

def render_xai_figures(perturbation_results, gradient_results, output_dir="gpu-server/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n[XAI] Generating publication-grade figures...")

    # Figure 4: Perturbation Feature Importance (LOSO) for 3 Clinical Cases
    fig, axes = plt.subplots(len(perturbation_results), 1, figsize=(10, 4 * len(perturbation_results)), sharex=False)
    if len(perturbation_results) == 1:
        axes = [axes]

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for idx, (res, ax) in enumerate(zip(perturbation_results, axes)):
        disease = res["baseline_prediction"]
        clauses = [a["clause"] for a in res["attributions"]][:6]
        drops = [a["marginal_drop"] * 100.0 for a in res["attributions"]][:6]

        # Truncate long clauses for clean visualization
        short_clauses = [c if len(c) <= 35 else c[:32] + "..." for c in clauses]

        y_pos = np.arange(len(short_clauses))
        bar_colors = ["#d9381e" if a["prediction_flipped"] else "#1f77b4" for a in res["attributions"][:6]]

        bars = ax.barh(y_pos, drops, color=bar_colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_clauses, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Confidence Drop on Ablation (Percentage Points)", fontsize=10, fontweight="bold")
        ax.set_title(f"Clinical Case {idx+1}: {disease} (Base Confidence: {res['baseline_probability']*100:.1f}%)\nRed Bar = Diagnosis Flipped upon Symptom Removal", fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:+.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p_fig_png = os.path.join(output_dir, "figure4_perturbation_attribution.png")
    p_fig_pdf = os.path.join(output_dir, "figure4_perturbation_attribution.pdf")
    fig.savefig(p_fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p_fig_png} and {p_fig_pdf}")

    # Figure 5: White-Box Gradient Token Saliency Comparison
    fig, axes = plt.subplots(len(gradient_results), 1, figsize=(10, 3.5 * len(gradient_results)))
    if len(gradient_results) == 1:
        axes = [axes]

    for idx, (res, ax) in enumerate(zip(gradient_results, axes)):
        tokens = [t["token"].strip() for t in res["top_salient_tokens"][:8]]
        scores = [t["input_x_grad"] for t in res["top_salient_tokens"][:8]]

        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, scores, color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Input x Gradient Attribution Score", fontsize=10, fontweight="bold")
        ax.set_title(f"White-Box Token Saliency: {res['predicted_disease']} (True: {res['true_label']})", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    g_fig_png = os.path.join(output_dir, "figure5_gradient_saliency.png")
    g_fig_pdf = os.path.join(output_dir, "figure5_gradient_saliency.pdf")
    fig.savefig(g_fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(g_fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_fig_png} and {g_fig_pdf}")

def generate_interactive_html(gradient_results, output_path="gpu-server/figures/xai_saliency_heatmaps.html"):
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Clinical Saliency Heatmaps - Fine-Tuned Clef-Flash</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; color: #212529; padding: 25px; line-height: 1.6; }
h1 { color: #0d6efd; margin-bottom: 5px; }
.card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 25px; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px; color: white; background: #198754; }
.saliency-box { background: #fafafa; border: 1px solid #dee2e6; border-radius: 6px; padding: 18px; font-size: 16px; line-height: 2.2; margin-top: 15px; }
.token { padding: 3px 6px; border-radius: 4px; margin: 0 2px; display: inline-block; }
.legend { margin-top: 15px; font-size: 13px; color: #6c757d; }
</style>
</head>
<body>
<h1>Clinical Explainability & Saliency Heatmaps</h1>
<p>Fine-Tuned <b>Clef-Flash</b> (9B Multimodal Decision Model) | White-Box Input-Gradient Attribution</p>
<hr style="border: 0; border-top: 1px solid #ccc; margin-bottom: 25px;">
"""
    for res in gradient_results:
        tokens_data = res["token_attributions"]
        # Normalize scores among word tokens
        valid_scores = [abs(t["input_x_grad"]) for t in tokens_data if len(t["token"].strip()) > 0]
        max_score = max(valid_scores) if valid_scores and max(valid_scores) > 0 else 1.0

        html += f"""
<div class="card">
  <h3>Case #{res['case_id']}: <span class="badge">{res['predicted_disease']}</span> <small style="color:#6c757d;font-weight:normal;">(True Diagnosis: {res['true_label']})</small></h3>
  <p><b>Original Narrative:</b> <i>"{res['patient_text']}"</i></p>
  <div class="saliency-box">
"""
        for t in tokens_data:
            tok_text = t["token"]
            score = t["input_x_grad"]
            # Color intensity 0 to 1
            intensity = min(1.0, max(0.0, abs(score) / max_score))
            if score > 0:
                bg = f"rgba(25, 135, 84, {0.1 + 0.7 * intensity:.2f})" # Green for supporting target
            else:
                bg = f"rgba(220, 53, 69, {0.1 + 0.7 * intensity:.2f})" # Red for opposing
            
            clean_tok = tok_text.replace("<", "&lt;").replace(">", "&gt;")
            if not clean_tok.strip().startswith("&lt;|"):
                html += f'<span class="token" style="background-color: {bg};" title="Score: {score:.4f}">{clean_tok}</span>'

        html += """
  </div>
  <div class="legend">🟢 Darker green indicates strong positive attribution toward the diagnosis.</div>
</div>
"""
    html += """
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved interactive HTML heatmap: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_clef_flash_lora_and_head.pt")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device = load_environment(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )

    # Curate high-impact test cases representing different clinical categories:
    # 1. Dermatological (Psoriasis / Acne)
    # 2. Respiratory (Bronchial Asthma / Pneumonia)
    # 3. Infectious / Tropical (Malaria / Dengue / Typhoid)
    test_cases = [
        {
            "id": 1,
            "label": "Psoriasis",
            "text": "I have an itchy red rash on my arms with dry silvery scales peeling off. My skin is cracked and bleeds occasionally."
        },
        {
            "id": 2,
            "label": "Bronchial Asthma",
            "text": "I've been struggling to breathe, coughing constantly, and coughing up a lot of thick mucoid sputum while feeling exhausted."
        },
        {
            "id": 3,
            "label": "Malaria",
            "text": "I have been experiencing severe shaking chills that recur every two days, followed by burning high fevers and drenching sweat."
        },
        {
            "id": 4,
            "label": "Arthritis",
            "text": "My neck has been extremely stiff and my muscles have been extremely weak with painful swelling in my joints making it difficult to walk."
        }
    ]

    # Run Method 1: Perturbation
    perturbation_results = run_perturbation_xai(
        model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases
    )

    # Run Method 2: White-Box Gradients
    gradient_results = run_whitebox_gradient_xai(
        model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases
    )

    # Render visualizations
    render_xai_figures(perturbation_results, gradient_results, output_dir=args.output_dir)
    generate_interactive_html(gradient_results, output_path=os.path.join(args.output_dir, "figure5_gradient_saliency_heatmaps.html"))

    # Export JSON logs
    with open(os.path.join(args.output_dir, "xai_perturbation_results.json"), "w", encoding="utf-8") as f:
        json.dump(perturbation_results, f, indent=2)

    with open(os.path.join(args.output_dir, "xai_gradient_results.json"), "w", encoding="utf-8") as f:
        json.dump(gradient_results, f, indent=2)

    print("\n[SUCCESS] Both XAI methods completed successfully and all artifacts saved!")

if __name__ == "__main__":
    main()
