#!/bin/bash
# Quick training script for testing/validation
# Uses optimal settings based on benchmarks

export HSA_OVERRIDE_GFX_VERSION=11.0.0

source venv/bin/activate

echo "Quick Training (10 epochs for testing)"
echo "========================================"

python3 src/train.py \
    --epochs 10 \
    --batch-size 8 \
    --lr 3e-4 \
    --audio-dir data/raw/fma_small \
    --wav-dir data/wav_cache \
    --use-memmap \
    --checkpoint-dir data/checkpoints \
    --num-workers 4 \
    --audio-length 32768

echo ""
echo "Quick training complete (~35-40 minutes)"
