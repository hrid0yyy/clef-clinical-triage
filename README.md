# Fine-Tuning Clef-Flash Against TF-IDF Baselines on Symptom2Disease

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2610.15029-b31b1b.svg)](https://arxiv.org/abs/2610.15029)

Code, data splits and saved results for the paper:

> **Is a 9B Decision Foundation Model Needed for Symptom-to-Disease Classification? Fine-Tuning Clef-Flash Against TF-IDF Baselines**
> S. M. Mahbub Uddin Ahmed, Department of Computer Science and Engineering, United International University
> arXiv:2610.15029

## Summary

[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) is a 9B decision foundation model: a Qwen3.5-9B backbone with a small transformer scoring head (JointSchemaHead) that scores a fixed set of candidate labels in one forward pass, without generating text. We fine-tune it with 4-bit QLoRA on the 24-class [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) dataset and compare it with classical baselines trained on **the same** 960/240 stratified split.

**Main finding:** fine-tuning lifts Clef-Flash from 65.00% to 95.42% accuracy, but TF-IDF linear models on the same split reach 98.75% with sub-millisecond CPU inference. On short, keyword-rich narratives like these, the 9B model offers no accuracy advantage over a well-tuned linear classifier.

## Results (240 test cases, same split)

| Model | Accuracy [95% Wilson CI] | Macro F1 | Latency per record |
| :--- | :---: | :---: | :---: |
| TF-IDF + linear SVM | **98.75%** [96.39, 99.57] | **0.9876** | 0.7 ms (laptop CPU) |
| TF-IDF + logistic regression | **98.75%** [96.39, 99.57] | **0.9876** | 0.8 ms (laptop CPU) |
| TF-IDF + random forest | 96.67% [93.56, 98.30] | 0.9665 | 45 ms (laptop CPU) |
| Clef 27B, zero-shot (API) | 66.25% [60.05, 71.93] | 0.6186 | 0.92 s (API, incl. network) |
| Clef-Flash 9B, zero-shot (API) | 62.92% [56.65, 68.78] | 0.5985 | 0.80 s (API, incl. network) |
| Clef-Flash 9B, zero-shot (local, 4-bit) | 65.00% [58.77, 70.76] | – | – |
| **Clef-Flash 9B, fine-tuned (QLoRA)** | 95.42% [91.98, 97.42] | 0.9551 | not measured |

The baselines' hyperparameters were chosen by 5-fold cross-validation on the training split only. There was no validation set for Clef-Flash: the checkpoint (epoch 4) was selected on test accuracy, so its figure may be slightly optimistic. Local inference latency of the fine-tuned model was not measured.

A Leave-One-Symptom-Out (LOSO) clause ablation on four illustrative, author-written narratives shows predictions depending on distinguishing clauses. For example, removing "dry silvery scales peeling off" from a psoriasis narrative lowers the psoriasis probability from above 99.99% to 0.12% and flips the prediction to impetigo. These are case studies, not a systematic evaluation.

## Repository structure

```
clef-clinical-triage/
├── configs/training_config.yaml      # QLoRA and optimizer hyperparameters
├── data/
│   ├── train_80_20.csv               # 960 training cases (40 per class)
│   ├── test_80_20.csv                # 240 test cases (10 per class)
│   ├── criteria.json                 # MedlinePlus-based criteria summaries (24 classes)
│   └── splits_metadata.json
├── src/
│   ├── dataset.py                    # Narrative -> decision schema encoding
│   ├── model.py                      # Clef-Flash loading, QLoRA adapters, JointSchemaHead
│   ├── train.py                      # Training loop (differential learning rates)
│   ├── evaluate.py                   # Accuracy, F1, confusion matrix
│   ├── inference.py                  # Single-narrative prediction
│   └── xai_engine.py                 # LOSO clause ablation (also contains an
│                                     #   Integrated Gradients routine not used in the paper)
├── baselines/
│   ├── run_classical_baselines.py    # TF-IDF SVM / LR / RF on the same split
│   └── classical_baselines_results.json
├── zero_shot/
│   ├── evaluate_clef_flash_test_split.py               # Clef-Flash via Cloudflare Workers AI
│   ├── evaluate_clef_test_split.py                     # Clef 27B via Cloudflare Workers AI
│   ├── evaluate_*_with_criteria.py                     # Same, with criteria text in the prompt
│   └── results/                                        # Saved per-case outputs and metrics
├── results/
│   ├── training_history.json         # Per-epoch loss and test accuracy
│   ├── evaluation_results.json       # Fine-tuned model: metrics and confusion matrix
│   └── xai_perturbation_results.json # LOSO ablation outputs (four narratives)
├── create_splits.py                  # Builds the stratified 80/20 split
├── run_xai_analysis.py               # Runs the LOSO analysis on a checkpoint
├── generate_paper_figures.py         # Training curves, confusion matrix, per-class F1
├── generate_figure_overview.py       # Overview figure (Fig. 1)
├── validate_data_and_schema.py       # Data and schema sanity checks
├── test_pipeline.py                  # Dataset encoding test
├── run_finetune.sh / run_finetune.bat
└── requirements.txt
```

## Reproducing the results

```bash
git clone https://github.com/hrid0yyy/clef-clinical-triage.git
cd clef-clinical-triage
pip install -r requirements.txt
python validate_data_and_schema.py
```

**Classical baselines (CPU, a few minutes):**
```bash
python baselines/run_classical_baselines.py
```
Latencies depend on the machine; accuracies are deterministic.

**Fine-tuning (one 24 GB GPU; about 35 minutes on an RTX 4090):**
```bash
python src/train.py --config configs/training_config.yaml
python src/evaluate.py --checkpoint checkpoints/best_clef_flash_model.pt
```

**LOSO clause ablation:**
```bash
python run_xai_analysis.py --checkpoint checkpoints/best_clef_flash_lora_and_head.pt
```

**Zero-shot API evaluation** (needs a Cloudflare account with Workers AI; run from the repository root):
```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
python zero_shot/evaluate_clef_flash_test_split.py
```

**Figures from the saved results:**
```bash
python generate_paper_figures.py
python generate_figure_overview.py
```

The fine-tuned checkpoint is not included because of its size.

## Limitations

Symptom2Disease is small (1,200 narratives of about 30 words), its provenance is undocumented, and it uses a closed set of 24 labels. Results come from one training run, no paired significance test against the baselines was possible, and the ablation analysis was not reviewed by clinicians. None of this code is intended for clinical use.

## Citation

```bibtex
@misc{ahmed2026clefflash,
  title  = {Is a 9B Decision Foundation Model Needed for Symptom-to-Disease Classification? Fine-Tuning Clef-Flash Against TF-IDF Baselines},
  author = {Ahmed, S. M. Mahbub Uddin},
  year   = {2026},
  eprint = {2610.15029},
  archivePrefix = {arXiv}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). The Clef and Clef-Flash weights are subject to Cloudflare's license on [Hugging Face](https://huggingface.co/Cloudflare/clef-flash). The Symptom2Disease data is from [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease).
