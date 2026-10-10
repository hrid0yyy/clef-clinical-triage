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

    with open(config["data"]["criteria_path"], "r", encoding="utf-8") as f:
        criteria_dict = json.load(f)
    classes = sorted(list(criteria_dict.keys()))

    model, processor, encode_record_fn, collate_records_fn, _ = load_clef_model_and_processor(
        model_id=config["model"]["model_id"],
        cache_dir=config["model"]["cache_dir"],
        device=device,
        config=config
    )

    model = setup_trainable_model(model, config)

    emb_weight_cached = None
    def robust_clef_forward_with_embeds(self, batch):
        nonlocal emb_weight_cached
        base_model = (
            self.language_model.get_base_model()
            if hasattr(self.language_model, "get_base_model")
            else self.language_model
        )
        media = batch.get("media") or {}
        text_model = base_model.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        
        inputs_embeds = batch.get("inputs_embeds", None)
        if inputs_embeds is not None:
            outputs = text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
                **media,
            )
        else:
            outputs = text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
                **media,
            )
        target_dtype = self.head.hidden_norm.weight.dtype if hasattr(self.head, "hidden_norm") else torch.bfloat16
        hidden_states = outputs.last_hidden_state.to(dtype=target_dtype)
        if emb_weight_cached is None or emb_weight_cached.dtype != target_dtype:
            emb_weight_cached = base_model.get_output_embeddings().weight.to(dtype=target_dtype)
        return self.head(
            hidden_states,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            emb_weight_cached,
        )

    model.__class__.forward = robust_clef_forward_with_embeds

    if os.path.exists(checkpoint_path):
        print(f"[XAI] Loading fine-tuned weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("[XAI] Checkpoint loaded successfully!")

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
        probs = F.softmax(logits.float(), dim=-1).squeeze(0).cpu().float().numpy()

    return probs, batch

def run_perturbation_xai(model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases):
    print("\n" + "="*70)
    print("  METHOD 1: PERTURBATION-BASED FEATURE ATTRIBUTION (LOSO)")
    print("="*70)

    results = []

    for case in test_cases:
        patient_text = case["text"]
        true_label = case.get("label", "Unknown")
        candidate_phrases = case.get("symptoms", [])

        print(f"\n--- Analyzing Case: True Label: [{true_label}] ---")
        print(f"Full Narrative: \"{patient_text}\"")

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

        # Fallback to regex splitting if phrases not pre-specified
        if not candidate_phrases:
            candidate_phrases = [c.strip() for c in re.split(r"[,.;]\s*", patient_text) if len(c.strip()) > 3]

        clause_attributions = []

        for phrase in candidate_phrases:
            ablated_text = re.sub(re.escape(phrase), "", patient_text, flags=re.IGNORECASE).replace("  ", " ").strip()
            if not ablated_text:
                continue

            ablated_probs, _ = predict_probabilities(
                model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, ablated_text
            )
            ablated_top1_idx = int(np.argmax(ablated_probs))
            ablated_top1_pred = classes[ablated_top1_idx]
            ablated_target_prob = float(ablated_probs[top1_idx])
            
            marginal_drop = top1_prob - ablated_target_prob
            pct_drop = (marginal_drop / max(top1_prob, 1e-6)) * 100.0

            alt_indices = np.argsort(ablated_probs)[::-1][:2]
            alt_surges = [{"disease": classes[i], "probability": float(ablated_probs[i])} for i in alt_indices if classes[i] != top1_pred]

            clause_attributions.append({
                "clause": phrase,
                "ablated_target_prob": ablated_target_prob,
                "marginal_drop": marginal_drop,
                "percent_confidence_drop": pct_drop,
                "prediction_flipped": (ablated_top1_pred != top1_pred),
                "new_top_prediction": ablated_top1_pred,
                "new_top_prob": float(ablated_probs[ablated_top1_idx]),
                "surging_alternatives": alt_surges
            })

            flip_str = f" [FLIPPED to {ablated_top1_pred} ({ablated_probs[ablated_top1_idx]*100:.1f}%)]" if ablated_top1_pred != top1_pred else ""
            print(f"  • Ablated: \"{phrase}\" -> Target Prob: {ablated_target_prob*100:.1f}% (Drop: {marginal_drop*100:+.2f}% / {pct_drop:.1f}% drop){flip_str}")

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

def run_whitebox_gradient_xai(model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases, ig_steps=12):
    print("\n" + "="*70)
    print("  METHOD 2: WHITE-BOX INTEGRATED GRADIENTS ATTRIBUTION")
    print("="*70)

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
        print(f"\n--- Integrated Gradients Analysis for: [{true_label}] ---")

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

        input_ids = batch["input_ids"] # (1, seq_len)
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Find the boundary of the patient narrative tokens inside the full input sequence
        # We search for the first and last token matching words from patient_text
        patient_words = set(re.findall(r"\b\w+\b", patient_text.lower()))
        patient_token_indices = []
        for idx, tok in enumerate(tokens):
            clean_t = tok.strip().lower()
            if any(w in clean_t or clean_t in w for w in patient_words) and len(clean_t) > 1:
                # Exclude tokens from system instruction
                if "which" not in clean_t and "disease" not in clean_t and "patient" not in clean_t and "instructions" not in clean_t:
                    patient_token_indices.append(idx)

        # 1. Base embeddings
        with torch.no_grad():
            base_embeds = embed_tokens(input_ids).detach() # (1, seq_len, dim)

        baseline_embeds = torch.zeros_like(base_embeds)

        # 2. Target logit
        with torch.no_grad():
            batch["inputs_embeds"] = base_embeds
            out_base = model(batch)
            logits_base = extract_logits(out_base)
            target_idx = torch.argmax(logits_base, dim=-1).item()
            target_disease = classes[target_idx]
            target_prob = F.softmax(logits_base.float(), dim=-1)[0, target_idx].item()

        print(f"Target Disease: {target_disease} (Prob: {target_prob*100:.2f}%)")

        # 3. Integrated Gradients
        accumulated_grads = torch.zeros_like(base_embeds)
        alphas = np.linspace(0.1, 1.0, ig_steps)

        for alpha in alphas:
            interp_embeds = (baseline_embeds + alpha * (base_embeds - baseline_embeds)).clone().detach().requires_grad_(True)
            batch["inputs_embeds"] = interp_embeds

            model.zero_grad()
            out = model(batch)
            logits = extract_logits(out)
            target_logit = logits[0, target_idx]
            target_logit.backward()

            if interp_embeds.grad is not None:
                accumulated_grads += interp_embeds.grad.detach()

            del out, logits, target_logit, interp_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_grads = accumulated_grads / ig_steps
        ig_tensor = (base_embeds - baseline_embeds) * avg_grads
        token_ig_scores = ig_tensor.sum(dim=-1).squeeze(0).float().cpu().numpy()

        batch.pop("inputs_embeds", None)

        # Focus token attributions on clinical narrative
        token_attributions = []
        patient_salient = []

        for i, (tok, tid, score) in enumerate(zip(tokens, input_ids[0].tolist(), token_ig_scores.tolist())):
            is_patient = i in patient_token_indices
            item = {
                "token_id": tid,
                "token": tok,
                "ig_score": float(score),
                "is_patient_narrative": is_patient
            }
            token_attributions.append(item)
            if is_patient and len(tok.strip()) > 1 and not tok.strip().startswith("<|"):
                patient_salient.append(item)

        patient_salient.sort(key=lambda x: abs(x["ig_score"]), reverse=True)

        print(f"Top-5 Clinical Symptoms (Patient Tokens):")
        for st in patient_salient[:5]:
            print(f"   Token: '{st['token'].strip()}' | IG Attribution: {st['ig_score']:+.4f}")

        results.append({
            "case_id": case.get("id", len(results) + 1),
            "true_label": true_label,
            "patient_text": patient_text,
            "predicted_disease": target_disease,
            "predicted_prob": target_prob,
            "token_attributions": token_attributions,
            "top_salient_tokens": patient_salient[:10]
        })

    return results

def render_xai_figures(perturbation_results, gradient_results, output_dir="gpu-server/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n[XAI] Rendering publication figures (300 DPI PNG & Vector PDF)...")

    # Figure 4: Perturbation Attribution (LOSO)
    num_cases = len(perturbation_results)
    fig, axes = plt.subplots(num_cases, 1, figsize=(11, 3.8 * num_cases), sharex=False)
    if num_cases == 1:
        axes = [axes]

    for idx, (res, ax) in enumerate(zip(perturbation_results, axes)):
        disease = res["baseline_prediction"]
        clauses = [a["clause"] for a in res["attributions"]][:5]
        drops = [a["marginal_drop"] * 100.0 for a in res["attributions"]][:5]
        short_clauses = [c if len(c) <= 38 else c[:35] + "..." for c in clauses]

        y_pos = np.arange(len(short_clauses))
        bar_colors = ["#d9381e" if a["prediction_flipped"] else "#1f77b4" for a in res["attributions"][:5]]

        bars = ax.barh(y_pos, drops, color=bar_colors, edgecolor="black", alpha=0.88, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_clauses, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Confidence Drop on Symptom Ablation (Percentage Points)", fontsize=11, fontweight="bold")
        
        status_note = " — Diagnosis Flipped to Alternative" if any(a["prediction_flipped"] for a in res["attributions"][:5]) else " — Essential Clinical Feature"
        ax.set_title(f"Case {idx+1}: {disease} (Base Probability: {res['baseline_probability']*100:.1f}%){status_note}", fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:+.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(6, 0),
                        textcoords="offset points",
                        ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p_fig_png = os.path.join(output_dir, "figure4_perturbation_attribution.png")
    p_fig_pdf = os.path.join(output_dir, "figure4_perturbation_attribution.pdf")
    fig.savefig(p_fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p_fig_png} and {p_fig_pdf}")

    # Figure 5: White-Box Integrated Gradients
    fig, axes = plt.subplots(len(gradient_results), 1, figsize=(11, 3.6 * len(gradient_results)))
    if len(gradient_results) == 1:
        axes = [axes]

    for idx, (res, ax) in enumerate(zip(gradient_results, axes)):
        tokens = [t["token"].strip() for t in res["top_salient_tokens"][:7]]
        scores = [t["ig_score"] for t in res["top_salient_tokens"][:7]]

        y_pos = np.arange(len(tokens))
        bar_colors = ["#2ca02c" if s >= 0 else "#d62728" for s in scores]
        bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor="black", alpha=0.88, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"'{t}'" for t in tokens], fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Integrated Gradients Attribution Score", fontsize=11, fontweight="bold")
        ax.set_title(f"Case {idx+1}: {res['predicted_disease']} — Clinical Token Attribution", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            offset = 6 if width >= 0 else -6
            ha = "left" if width >= 0 else "right"
            ax.annotate(f"{width:+.3f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(offset, 0),
                        textcoords="offset points",
                        ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    g_fig_png = os.path.join(output_dir, "figure5_gradient_saliency.png")
    g_fig_pdf = os.path.join(output_dir, "figure5_gradient_saliency.pdf")
    fig.savefig(g_fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(g_fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_fig_png} and {g_fig_pdf}")

def generate_interactive_html(gradient_results, output_path="gpu-server/figures/figure5_gradient_saliency_heatmaps.html"):
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
<p>Fine-Tuned <b>Clef-Flash</b> (9B Multimodal Decision Model) | White-Box Integrated Gradients Attribution</p>
<hr style="border: 0; border-top: 1px solid #ccc; margin-bottom: 25px;">
"""
    for res in gradient_results:
        tokens_data = res["token_attributions"]
        # Only take patient tokens
        patient_tokens = [t for t in tokens_data if t.get("is_patient_narrative", False)]
        if not patient_tokens:
            patient_tokens = tokens_data

        valid_scores = [abs(t["ig_score"]) for t in patient_tokens if len(t["token"].strip()) > 0]
        max_score = max(valid_scores) if valid_scores and max(valid_scores) > 0 else 1.0

        html += f"""
<div class="card">
  <h3>Case #{res['case_id']}: <span class="badge">{res['predicted_disease']}</span> <small style="color:#6c757d;font-weight:normal;">(True Diagnosis: {res['true_label']})</small></h3>
  <p><b>Original Narrative:</b> <i>"{res['patient_text']}"</i></p>
  <div class="saliency-box">
"""
        for t in patient_tokens:
            tok_text = t["token"]
            score = t["ig_score"]
            intensity = min(1.0, max(0.0, abs(score) / max_score))
            if score >= 0:
                bg = f"rgba(25, 135, 84, {0.1 + 0.7 * intensity:.2f})"
            else:
                bg = f"rgba(220, 53, 69, {0.1 + 0.7 * intensity:.2f})"
            
            clean_tok = tok_text.replace("<", "&lt;").replace(">", "&gt;")
            if not clean_tok.strip().startswith("&lt;|"):
                html += f'<span class="token" style="background-color: {bg};" title="IG Score: {score:+.4f}">{clean_tok}</span>'

        html += """
  </div>
  <div class="legend">🟢 Green indicates positive feature attribution toward the diagnosis (Integrated Gradients).</div>
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

    test_cases = [
        {
            "id": 1,
            "label": "Psoriasis",
            "text": "I have an itchy red rash on my arms with dry silvery scales peeling off. My skin is cracked and bleeds occasionally.",
            "symptoms": [
                "itchy red rash",
                "dry silvery scales peeling off",
                "skin is cracked",
                "bleeds occasionally"
            ]
        },
        {
            "id": 2,
            "label": "Bronchial Asthma",
            "text": "I've been struggling to breathe, coughing constantly, and coughing up a lot of thick mucoid sputum while feeling exhausted.",
            "symptoms": [
                "struggling to breathe",
                "coughing constantly",
                "coughing up a lot of thick mucoid sputum",
                "feeling exhausted"
            ]
        },
        {
            "id": 3,
            "label": "Malaria",
            "text": "I have been experiencing severe shaking chills that recur every two days, followed by burning high fevers and drenching sweat.",
            "symptoms": [
                "severe shaking chills",
                "recur every two days",
                "burning high fevers",
                "drenching sweat"
            ]
        },
        {
            "id": 4,
            "label": "Arthritis",
            "text": "My neck has been extremely stiff and my muscles have been extremely weak with painful swelling in my joints making it difficult to walk.",
            "symptoms": [
                "neck has been extremely stiff",
                "muscles have been extremely weak",
                "painful swelling in my joints",
                "difficult to walk"
            ]
        }
    ]

    perturbation_results = run_perturbation_xai(
        model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases
    )

    gradient_results = run_whitebox_gradient_xai(
        model, processor, encode_record_fn, collate_records_fn, criteria_dict, classes, config, device, test_cases, ig_steps=6
    )

    render_xai_figures(perturbation_results, gradient_results, output_dir=args.output_dir)
    generate_interactive_html(gradient_results, output_path=os.path.join(args.output_dir, "figure5_gradient_saliency_heatmaps.html"))

    with open(os.path.join(args.output_dir, "xai_perturbation_results.json"), "w", encoding="utf-8") as f:
        json.dump(perturbation_results, f, indent=2)

    with open(os.path.join(args.output_dir, "xai_gradient_results.json"), "w", encoding="utf-8") as f:
        json.dump(gradient_results, f, indent=2)

    print("\n[SUCCESS] Both XAI methods completed successfully and all artifacts saved!")

if __name__ == "__main__":
    main()
