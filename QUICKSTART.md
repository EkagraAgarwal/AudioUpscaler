# Quick Start Guide - Optimized Training

Get started quickly with optimal performance settings for AMD RX 7700S.

## Prerequisites

1. **Convert audio to WAV** (one-time setup):
   ```bash
   python3 convert_to_wav.py --src data/raw/fma_small --dst data/wav_cache
   ```

2. **Verify conversion**:
   ```bash
   python3 convert_to_wav.py --verify --dst data/wav_cache
   ```

## Quick Training (10 epochs - ~40 minutes)

Test your setup quickly:
```bash
./run_quick_training.sh
```

Or manually:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
source venv/bin/activate

python3 src/train.py \
    --epochs 10 \
    --batch-size 8 \
    --use-memmap \
    --wav-dir data/wav_cache \
    --audio-dir data/raw/fma_small
```

## Standard Training (50 epochs - ~2 hours)

Best balance of quality and time:
```bash
./run_training.sh
```

## Full Training (100 epochs - ~6-7 hours)

Maximum quality:
```bash
./run_full_training.sh
```

## Optimal Settings (Based on Benchmarks)

Your configuration is already optimized:

| Setting | Value | Reason |
|---------|-------|--------|
| **batch_size** | 8 | Optimal for 8GB VRAM |
| **num_workers** | 4 | Optimal for 16-core CPU |
| **use_memmap** | true | 15-20% speedup |
| **audio_length** | 32768 | Fast training, good context |
| **STFT scales** | 3 | Reducing provides <1% gain |

## Performance

- **Speed**: ~3.5-4.0 min/epoch
- **Memory**: ~2-3 GB VRAM usage
- **CPU**: 4 workers (out of 16 cores)

## What NOT to Change

Based on benchmarks, these settings are **already optimal**:

❌ **Don't increase batch size** - Actually slows down (0.54x)
❌ **Don't add more workers** - Adds overhead (0.92x)
❌ **Don't reduce STFT scales** - Minimal gain (<1%)
❌ **Can't use mixed precision** - ROCm 6.3 incompatible
❌ **Can't use torch.compile()** - Requires ROCm 7.x

## Monitoring

Watch training progress:
```bash
tensorboard --logdir runs
```

Open http://localhost:6006 in browser.

## Troubleshooting

### GPU not detected
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Slow training
- Ensure `--use-memmap` flag is set
- Verify WAV files exist: `ls data/wav_cache/*.wav | wc -l`
- Check GPU utilization: `rocm-smi`

### Out of memory
- Reduce audio_length: `--audio-length 16384`
- Reduce batch_size: `--batch-size 4`

## Next Steps

1. Run quick training to verify setup
2. Monitor with TensorBoard
3. Run full training (50-100 epochs)
4. Evaluate model quality
5. Adjust epochs as needed

## Performance Comparison

| Configuration | Time/Epoch | Total (50 epochs) |
|---------------|-----------|-------------------|
| MP3 (baseline) | 4.5 min | 3.75 hours |
| **MemMap WAV** | **3.5 min** | **3.0 hours** |
| MemMap + Quick | 2.5 min | 2.0 hours |

**Recommendation**: Use MemMap WAV configuration for best results.
