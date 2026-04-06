#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

source venv/bin/activate

echo "========================================"
echo "Starting Training on AMD GPU"
echo "========================================"
echo "HSA Override: $HSA_OVERRIDE_GFX_VERSION"
echo "GPU Device: $HIP_VISIBLE_DEVICES"
echo ""

python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Device Count: {torch.cuda.device_count()}')
print()
"

python3 src/train.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 3e-4 \
    --audio-dir data/raw/fma_small \
    --checkpoint-dir data/checkpoints \
    --num-workers 4 \
    --audio-length 65536 \
    --sample-rate 44100 \
    --bitrate 128

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
