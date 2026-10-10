#!/bin/bash
set -e

echo "======================================================================"
echo "Clef-Flash Clinical Fine-Tuning Pipeline (Symptom2Disease 80/20 Split)"
echo "======================================================================"

# 1. Check GPU Availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Training may run on CPU (very slow)."
fi

# 2. Install Dependencies
echo ""
echo "Installing required Python packages..."
pip install -r requirements.txt

# 3. Launch Training
echo ""
echo "Starting fine-tuning process..."
python src/train.py --config configs/training_config.yaml

# 4. Evaluate Best Checkpoint
echo ""
echo "Running post-training evaluation on 240 test cases..."
python src/evaluate.py --checkpoint checkpoints/best_clef_flash_model.pt --config configs/training_config.yaml

echo ""
echo "Pipeline finished successfully!"
