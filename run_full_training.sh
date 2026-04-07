#!/bin/bash
# Long training script for best quality model
# 100 epochs for maximum quality

export HSA_OVERRIDE_GFX_VERSION=11.0.0

source venv/bin/activate

echo "========================================"
echo "Full Training (100 epochs)"
echo "========================================"
echo "Expected time: 6-7 hours"
echo "Start time: $(date)"
echo ""

python3 src/train.py \
    --epochs 100 \
    --batch-size 8 \
    --lr 3e-4 \
    --audio-dir data/raw/fma_small \
    --wav-dir data/wav_cache \
    --use-memmap \
    --checkpoint-dir data/checkpoints \
    --num-workers 4 \
    --audio-length 32768

echo ""
echo "========================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "========================================"
