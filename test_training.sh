#!/bin/bash

# Quick test script - train for just 1 epoch to verify everything works
echo "Quick training test (1 epoch)..."

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export MIOPEN_LOG_LEVEL=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HSA_ENABLE_SDMA=0
export MIOPEN_FIND_MODE=normal
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache
export MIOPEN_SYSTEM_DB_PATH=/tmp/miopen-cache

mkdir -p /tmp/miopen-cache

source venv/bin/activate

python src/train.py \
    --use-memmap \
    --batch-size 8 \
    --num-workers 4 \
    --audio-length 32768 \
    --dynamic-bitrate \
    --epochs 1 \
    --lr 3e-4

echo ""
echo "Test complete! If successful, run full training with:"
echo "  ./train_optimized.sh"
