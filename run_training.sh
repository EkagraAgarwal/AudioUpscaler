#!/bin/bash
# Optimal training configuration for AMD RX 7700S (ROCm 7.x)
# Updated with improved architecture for better audio quality

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export MIOPEN_LOG_LEVEL=4

source venv/bin/activate

echo "========================================"
echo "Optimized Training on AMD RX 7700S"
echo "========================================"
echo "HSA Override: $HSA_OVERRIDE_GFX_VERSION"
echo "GPU Device: $HIP_VISIBLE_DEVICES"
echo ""
echo "Architecture Improvements:"
echo " - Base channels: 48 (11.7M parameters)"
echo " - Residual connections in encoder/decoder"
echo " - Global residual connection (input + output)"
echo " - Dilated bottleneck (rates: 1, 2, 4, 8, 16)"
echo " - Interpolation upsampling (no checkerboard artifacts)"
echo " - Spectral convergence + log-magnitude STFT loss"
echo " - Dynamic loss weighting (L1 heavy early: 20x)"
echo " - ReduceLROnPlateau scheduler"
echo ""
echo "Optimal Settings:"
echo " - Batch size: 12 (balanced for larger model)"
echo " - Num workers: 4"
echo " - Audio length: 32768"
echo " - Dynamic bitrate: 32-128 kbps (curriculum learning)"
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
  --batch-size 12 \
  --lr 3e-4 \
  --audio-dir data/raw/fma_small \
  --wav-dir data/wav_cache \
  --use-memmap \
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
echo "Architecture Improvements Applied:"
echo " - 11.7M parameters (base_channels=48)"
echo " - Residual connections for better gradient flow"
echo " - Global skip connection preserves input structure"
echo " - Dilated convolutions expand receptive field"
echo " - Interpolation upsampling prevents artifacts"
echo " - Spectral convergence improves phase alignment"
echo " - Dynamic loss weighting: 20x L1 early, balanced later"
echo " - ReduceLROnPlateau prevents plateau at epoch 50"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir runs"
echo ""
echo "To run inference:"
echo "  python3 src/inference.py --input <audio.wav> --checkpoint data/checkpoints/best_model.pt"
echo ""
