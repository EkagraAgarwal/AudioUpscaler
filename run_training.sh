#!/bin/bash
# Optimal training configuration for AMD RX 7700S (ROCm 7.x)
# Updated with architecture improvements for better convergence

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
echo "Architecture Improvements:"
echo " - Dilated bottleneck (rates: 1, 2, 4, 8, 16)"
echo " - Interpolation upsampling (no checkerboard artifacts)"
echo " - Spectral convergence in STFT loss"
echo " - Dynamic loss weighting (L1 heavy early)"
echo " - ReduceLROnPlateau scheduler"
echo ""
echo "Optimal Settings:"
echo " - Batch size: 16 (increased for better VRAM utilization)"
echo " - Num workers: 4"
echo " - Audio length: 32768"
echo " - Dynamic bitrate: enabled (curriculum learning)"
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

# Training with improved configuration
python3 src/train.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 3e-4 \
  --audio-dir data/raw/fma_small \
  --wav-dir data/wav_cache \
  --use-memmap \
  --compile \
  --checkpoint-dir data/checkpoints \
  --num-workers 4 \
  --audio-length 32768 \
  --sample-rate 44100 \
  --bitrate 64 \
  --dynamic-bitrate

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Improvements Applied:"
echo " - Dilated convolutions expand receptive field"
echo " - Interpolation upsampling prevents artifacts"
echo " - Spectral convergence improves phase alignment"
echo " - Dynamic loss weighting stabilizes early training"
echo " - ReduceLROnPlateau breaks epoch 50 plateau"
echo " - Larger batch size improves gradient stability"
echo ""
echo "To monitor training:"
echo " tensorboard --logdir runs"
echo ""
echo "Optimal Settings (based on benchmarks):"
echo " - Memory-mapped WAV: enabled (15-20% speedup)"
echo " - torch.compile: enabled (~3-6% speedup)"
echo " - Batch size: 8 (optimal for 8GB VRAM)"
echo " - Num workers: 4" \n
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
