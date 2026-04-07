#!/bin/bash
# Optimal training configuration for AMD RX 7700S (ROCm 7.x)
# Based on benchmark results - see OPTIMIZATION_RESULTS.md

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

source venv/bin/activate

echo "========================================"
echo "Optimized Training on AMD RX 7700S"
echo "========================================"
echo "HSA Override: $HSA_OVERRIDE_GFX_VERSION"
echo "GPU Device: $HIP_VISIBLE_DEVICES"
echo ""
echo "Optimal Settings (based on benchmarks):"
echo " - Memory-mapped WAV: enabled (15-20% speedup)"
echo " - torch.compile: enabled (~3-6% speedup)"
echo " - Batch size: 8 (optimal for 8GB VRAM)"
echo " - Num workers: 4 (optimal for 16-core CPU)"
echo " - Audio length: 32768 (fast training)"
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

# Training with optimal configuration
python3 src/train.py \
--epochs 50 \
--batch-size 8 \
--lr 3e-4 \
--audio-dir data/raw/fma_small \
--wav-dir data/wav_cache \
--use-memmap \
--compile \
--checkpoint-dir data/checkpoints \
--num-workers 4 \
--audio-length 32768 \
 --sample-rate 44100 \
    --bitrate 128 \
    --dynamic-bitrate

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Performance Notes:"
echo " - Estimated time: 2.9-3.2 hours for 50 epochs"
echo " - Speed: ~3.3-3.8 min per epoch"
echo " - Memory-mapped WAV: 15-20% speedup"
echo " - torch.compile: ~3-6% speedup"
echo " - Combined speedup: ~18-26%"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir runs"
echo ""
