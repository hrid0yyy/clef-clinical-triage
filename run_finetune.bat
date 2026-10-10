@echo off
echo ======================================================================
echo Clef-Flash Clinical Fine-Tuning Pipeline (Symptom2Disease 80/20 Split)
echo ======================================================================

where nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo GPU detected:
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
) else (
    echo WARNING: nvidia-smi not found. Training may run on CPU.
)

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Launching fine-tuning...
python src\train.py --config configs\training_config.yaml

echo.
echo Evaluating best checkpoint...
python src\evaluate.py --checkpoint checkpoints\best_clef_flash_model.pt --config configs\training_config.yaml

echo.
echo Done!
pause
