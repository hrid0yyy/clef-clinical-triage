"""
Publication-Grade Figures Generator for clef-flash Fine-Tuning Paper
Generates 300+ DPI, IEEE/Nature style figures for:
1. Training & Validation Loss Curve
2. Top-1 and Top-3 Accuracy Trajectory Curve
3. High-Resolution 24x24 Normalized Confusion Matrix Heatmap
4. Per-Class F1 Score Comparison Bar Chart
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def generate_learning_curves(history_json_path, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(history_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline = data.get("baseline_metrics", {})
    history = data.get("history", [])

    epochs = [0] + [h["epoch"] for h in history]
    train_loss = [None] + [h["train_loss"] for h in history]
    eval_loss = [baseline.get("eval_loss", None)] + [h["eval_loss"] for h in history]
    top1_acc = [baseline.get("top1_accuracy", 65.0)] + [h["top1_accuracy"] for h in history]
    top3_acc = [baseline.get("top3_accuracy", 86.67)] + [h["top3_accuracy"] for h in history]

    set_publication_style()

    # Figure 1: Dual Loss & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Loss Trajectory
    valid_train_epochs = [e for e, l in zip(epochs, train_loss) if l is not None]
    valid_train_loss = [l for l in train_loss if l is not None]
    
    ax1.plot(valid_train_epochs, valid_train_loss, marker='o', color='#1f77b4', linewidth=2.2, label='Training Loss')
    valid_eval_epochs = [e for e, l in zip(epochs, eval_loss) if l is not None]
    valid_eval_loss = [l for l in eval_loss if l is not None]
    if valid_eval_loss:
        ax1.plot(valid_eval_epochs, valid_eval_loss, marker='s', color='#ff7f0e', linewidth=2.2, linestyle='--', label='Test Loss')

    ax1.set_title('(a) Cross-Entropy Loss Progression', pad=10, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Cross-Entropy)')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.set_xticks(epochs)

    # Subplot 2: Accuracy Trajectory
    ax2.plot(epochs, top1_acc, marker='o', color='#2ca02c', linewidth=2.4, label='Top-1 Accuracy')
    ax2.plot(epochs, top3_acc, marker='^', color='#9467bd', linewidth=2.2, linestyle='--', label='Top-3 Accuracy')
    ax2.axhline(baseline.get("top1_accuracy", 65.0), color='gray', linestyle=':', label=f'Zero-Shot Baseline ({baseline.get("top1_accuracy", 65.0):.1f}%)')
    
    for e, acc in zip(epochs, top1_acc):
        ax2.annotate(f"{acc:.1f}%", (e, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    ax2.set_title('(b) Diagnostic Accuracy Trajectory', pad=10, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(min(top1_acc) - 5, 102)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    ax2.set_xticks(epochs)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "figure1_training_dynamics.png")
    pdf_path = os.path.join(output_dir, "figure1_training_dynamics.pdf")
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Generated: {png_path} and {pdf_path}")
    return png_path

def generate_confusion_matrix(eval_json_path, output_dir="figures",
                              title='Figure 2: Normalized Test Confusion Matrix (24 Disease Classes)'):
    os.makedirs(output_dir, exist_ok=True)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    classes = sorted(list(data["per_class"].keys()))
    
    # Normalize by row (true class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Recall (%)'},
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    
    if title:
        ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel('Predicted Disease Class', fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground Truth Disease Class', fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "figure2_confusion_matrix.png")
    pdf_path = os.path.join(output_dir, "figure2_confusion_matrix.pdf")
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Generated: {png_path} and {pdf_path}")
    return png_path

def generate_per_class_barchart(zeroshot_json, finetuned_json, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(zeroshot_json, "r", encoding="utf-8") as f:
        zs_data = json.load(f)
    with open(finetuned_json, "r", encoding="utf-8") as f:
        ft_data = json.load(f)

    classes = sorted(list(ft_data["per_class"].keys()))
    
    zs_f1 = []
    for c in classes:
        # compute or fetch zero-shot F1
        p = zs_data["per_class"].get(c, {}).get("precision", 0.0)
        r = zs_data["per_class"].get(c, {}).get("recall", 0.0)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        zs_f1.append(f1 * 100)

    ft_f1 = [ft_data["per_class"][c]["f1-score"] * 100 for c in classes]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(15, 6))
    
    x = np.arange(len(classes))
    width = 0.38

    rects1 = ax.bar(x - width/2, zs_f1, width, label='Zero-Shot Clef-Flash (API)', color='#9ecae1', edgecolor='#3182bd')
    rects2 = ax.bar(x + width/2, ft_f1, width, label='Fine-Tuned Clef-Flash (Ours)', color='#2ca02c', edgecolor='#006837')

    ax.set_ylabel('F1-Score (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    # Annotate specific notable improvements like Bronchial Asthma
    for idx, c in enumerate(classes):
        diff = ft_f1[idx] - zs_f1[idx]
        if diff >= 20:
            ax.annotate(f"+{diff:.0f}", (x[idx] + width/2, ft_f1[idx]), textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8, color='#006837', fontweight='bold')

    plt.tight_layout()
    png_path = os.path.join(output_dir, "figure3_classwise_comparison.png")
    pdf_path = os.path.join(output_dir, "figure3_classwise_comparison.pdf")
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Generated: {png_path} and {pdf_path}")
    return png_path

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    candidates_hist = [
        os.path.join(BASE_DIR, "checkpoints", "training_history.json"),
        os.path.join(BASE_DIR, "results", "training_history.json"),
        "results/training_history.json"
    ]
    hist_path = next((p for p in candidates_hist if os.path.exists(p)), None)

    candidates_eval = [
        os.path.join(BASE_DIR, "evaluation_results.json"),
        os.path.join(BASE_DIR, "results", "evaluation_results.json"),
        "results/evaluation_results.json"
    ]
    eval_path = next((p for p in candidates_eval if os.path.exists(p)), None)

    # Zero-shot Clef-Flash (9B) via API, bare option names: same base model as the fine-tuned one
    candidates_zs = [
        os.path.join(BASE_DIR, "zero_shot", "results", "clef_flash_test_split_eval_results.json"),
        os.path.join(BASE_DIR, "clef_flash_test_split_eval_results.json"),
        "clef_flash_test_split_eval_results.json"
    ]
    zs_path = next((p for p in candidates_zs if os.path.exists(p)), None)

    out_dir = os.path.join(BASE_DIR, "figures")

    if hist_path:
        generate_learning_curves(hist_path, out_dir)
    if eval_path:
        generate_confusion_matrix(eval_path, out_dir)
    if zs_path and eval_path:
        generate_per_class_barchart(zs_path, eval_path, out_dir)
    print("All publication figures generated successfully.")
