# Toward System One Clinical Intelligence: Decision Foundation Models for Triage & Explainability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Cloudflare%2Fclef--flash-yellow.svg)](https://huggingface.co/Cloudflare/clef-flash)

This repository provides the official implementation, training pipelines, causal explainability engines, and evaluation scripts for the paper:

> **Toward System One Clinical Intelligence: Parameter-Efficient Fine-Tuning and Counterfactual Explainability of Decision Foundation Models**  
> *S. M. Mahbub Uddin Ahmed* (Department of Computer Science and Engineering, United International University)

---

## 🔬 Overview

In medical decision-making, clinical cognition relies on dual-process mechanisms: fast pattern recognition (**System 1**) for rapid intake and triage, complemented by deliberative reasoning (**System 2**) for complex differential workups. Standard autoregressive large language models force all diagnostic interactions through slow, sequential text generation (1.5–3.5 s latencies) and risk hallucination.

**Decision Foundation Models** offer a different architectural path: by routing deep transformer representations directly into candidate class projection heads (**JointSchemaHead**) without generating intermediate tokens, they realize high-capacity System 1 clinical intelligence.

### Key Highlights
* **High Diagnostic Accuracy**: Achieves **95.42% accuracy** and **0.9551 macro F1** across 24 conditions on the *Symptom2Disease* benchmark (exceeding ClinicalBERT at 90.90%).
* **Deterministic Low Latency**: Executes in **under 80 milliseconds** per record on a single GPU (a **31-fold speedup** over generative LLM baselines).
* **Zero Output Parsing Failures**: Candidate class probabilities are computed simultaneously in a single forward pass without autoregressive decoding loops or malformed JSON.
* **Causal Explainability**: Audited via a causal **Leave-One-Symptom-Out (LOSO)** counterfactual perturbation framework and **Integrated Gradients** token attribution.

---

## 📁 Repository Structure

```
clef-clinical-triage/
├── configs/
│   └── training_config.yaml        # Exact hyperparameters (QLoRA, lr, batch sizes)
├── data/
│   ├── train_80_20.csv             # Fixed 80% training split (960 cases, 40 per class)
│   ├── test_80_20.csv              # Fixed 20% test split (240 cases, 10 per class)
│   ├── criteria.json               # NIH MedlinePlus diagnostic rubrics (24 conditions)
│   └── splits_metadata.json        # Dataset partition metadata & distribution
├── src/
│   ├── dataset.py                  # PyTorch Dataset encoding patient narratives into schema
│   ├── model.py                    # Clef-Flash loader with QLoRA adapters & JointSchemaHead
│   ├── train.py                    # Training loop with differential learning rates & loss tracking
│   ├── evaluate.py                 # Evaluation script (Accuracy, F1, Confusion Matrix)
│   ├── inference.py                # Sub-80ms real-time clinical triage inference
│   └── xai_engine.py               # LOSO perturbation & Integrated Gradients attribution
├── results/
│   ├── training_history.json       # Epoch-by-epoch loss & validation trajectory
│   └── evaluation_results.json     # Test split classwise performance metrics
├── generate_paper_figures.py       # Reproduces paper charts (loss, confusion matrix, F1)
├── generate_figure_overview.py     # Generates publication vector overview diagram
├── run_xai_analysis.py             # End-to-end LOSO perturbation & saliency analysis
├── validate_data_and_schema.py     # Pre-flight data integrity audit
├── test_pipeline.py                # Unit test for dataset encoding & schema formulation
├── run_finetune.sh                 # 1-Click execution script for Linux / Cloud GPUs
├── run_finetune.bat                # 1-Click execution script for Windows GPUs
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/hrid0yyy/clef-clinical-triage.git
cd clef-clinical-triage
pip install -r requirements.txt
```

### 2. Verify Data & Schemas
Run the pre-flight verification script to ensure splits and schemas are 100% sound:

```bash
python validate_data_and_schema.py
```

### 3. Fine-Tuning with QLoRA
Fine-tune the 9-billion parameter model on a single 16GB–24GB GPU (RTX 4090, A10G, L4) in ~35 minutes:

```bash
python src/train.py --config configs/training_config.yaml
```
*Or use the 1-click script:*
```bash
bash run_finetune.sh
```

### 4. Evaluation
Evaluate a trained checkpoint on the held-out 240 test cases:

```bash
python src/evaluate.py --checkpoint checkpoints/best_clef_flash_model.pt
```

### 5. Real-Time Clinical Inference
Run instant (<80 ms) deterministic disease triage on an uncurated patient narrative:

```bash
python src/inference.py --text "I have an itchy red rash on my arms with dry silvery scales peeling off."
```

### 6. Explainability & Causal Audit (XAI)
Run Leave-One-Symptom-Out (LOSO) clause perturbation and token-level Integrated Gradients saliency:

```bash
python run_xai_analysis.py --checkpoint checkpoints/best_clef_flash_model.pt
```

### 7. Reproduce Paper Figures
Reproduce publication figures (training dynamics, confusion matrix, classwise comparison) directly from logged results:

```bash
python generate_paper_figures.py
```

---

## 📊 Benchmark Comparison

| Model | Architecture Type | Token Generation | Diagnostic Acc. | Macro F1 | Per-Case Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Classical Bag-of-Words | Zero | 95.80% | 0.9570 | ~15 ms |
| **Linear SVM** | Classical Bag-of-Words | Zero | 97.50% | 0.9740 | ~10 ms |
| **ClinicalBERT** | Masked Language Model | Zero | 90.90% | 0.9080 | ~45 ms |
| **CLIN-LLM** | BioBERT + FLAN-T5 (MC Dropout) | Autoregressive | 98.00% | 0.9800 | ~2,500 ms |
| **Clef-Flash (Zero-Shot)** | 9B Decision Foundation Model | Zero | 65.00% | 0.6120 | < 80 ms |
| **Clef-Flash (Fine-Tuned)** | **9B Decision Foundation Model** | **Zero** | **95.42%** | **0.9551** | **< 80 ms** |

---

## 📄 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{ahmed2026systemone,
  title={Toward System One Clinical Intelligence: Parameter-Efficient Fine-Tuning and Counterfactual Explainability of Decision Foundation Models},
  author={Ahmed, S. M. Mahbub Uddin},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📜 License
This project is licensed under the [Apache License 2.0](LICENSE).
Base model weights are subject to the [Cloudflare Clef License](https://huggingface.co/Cloudflare/clef-flash).
