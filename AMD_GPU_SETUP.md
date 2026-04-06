# AMD GPU Setup Guide

This guide explains how to set up and train the audio upscaler model on AMD GPUs using ROCm.

## Prerequisites

- AMD GPU with ROCm support (tested on RX 7700S, RDNA3 architecture)
- ROCm 7.2.1 installed (or compatible version)
- Python 3.10+
- 8+ GB RAM
- 20+ GB disk space

## Installation

### 1. Install ROCm

Follow the official ROCm installation guide for your system:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

### 2. Set up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with ROCm Support

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Required for RDNA3
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
pip install pydub  # Required for MP3 loading
pip install -r requirements.txt
```

### 4. Verify GPU Detection

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
CUDA available: True
Device: AMD Radeon™ RX 7700S
```

## Training

### Quick Start

```bash
./run_training.sh
```

### Manual Training

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

python3 src/train.py \
    --epochs 50 \
    --batch-size 8 \
    --audio-dir data/raw/fma_small \
    --checkpoint-dir data/checkpoints \
    --num-workers 4 \
    --audio-length 65536
```

## Performance

On AMD RX 7700S:
- **Batch size**: 8
- **Audio length**: 65536 samples (~1.5 seconds at 44.1kHz)
- **Speed**: ~1.7 iterations/second
- **Time per epoch**: ~4.5 minutes
- **Recommended epochs**: 30-50
- **Total training time**: 2-4 hours

## Troubleshooting

### GPU Not Detected

**Issue**: `CUDA available: False`

**Solution**:
1. Check user groups:
```bash
groups
# Should include: video, render
```

2. Add user to groups (if missing):
```bash
sudo usermod -aG video,render $USER
# Logout and login again
```

3. Check device permissions:
```bash
ls -la /dev/kfd /dev/dri/
```

4. Set environment variable:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RDNA3
```

### Permission Denied on /dev/kfd

**Issue**: `Unable to open /dev/kfd read-write: Permission denied`

**Solution**:
```bash
# Check if user is in render group
groups

# If not, add user
sudo usermod -aG render $USER
# Logout and login again

# Alternative: Run with sg
sg render -c "python3 src/train.py ..."
```

### Out of Memory

**Issue**: CUDA out of memory errors

**Solution**:
1. Reduce batch size:
```bash
python src/train.py --batch-size 4  # or 2
```

2. Reduce audio length:
```bash
python src/train.py --audio-length 32768
```

3. Reduce number of workers:
```bash
python src/train.py --num-workers 2
```

### Slow Training

**Issue**: Training is slower than expected

**Possible causes**:
1. **Data loading bottleneck**: Audio files are loaded on-demand
2. **CPU bottleneck**: Data augmentation on CPU
3. **Small batch size**: Not utilizing GPU fully

**Solutions**:
- Use memory-mapped WAV files (see Option 3 in docs)
- Increase batch size if GPU memory allows
- Use multiple data loader workers

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Required for RDNA3 GPUs (RX 7000 series) |
| `HIP_VISIBLE_DEVICES` | `0` | Select specific GPU |
| `ROCR_VISIBLE_DEVICES` | `0` | Alternative GPU selection |

## Supported AMD GPUs

| Architecture | GFX Version | Example GPUs |
|-------------|-------------|--------------|
| RDNA3 | 11.0.0 | RX 7600, 7700S, 7800 XT, 7900 XTX |
| RDNA2 | 10.3.0 | RX 6600, 6700 XT, 6800 XT, 6900 XT |
| CDNA | 9.0.0 | MI100, MI200, MI250X |

For other architectures, adjust `HSA_OVERRIDE_GFX_VERSION` accordingly.

## PyTorch ROCm Compatibility

This setup uses:
- **PyTorch**: 2.10.0.dev (nightly)
- **ROCm**: 6.3 compatible
- **TorchCodec**: Not compatible (requires CUDA)

Audio loading via `pydub` instead of `torchaudio` to avoid TorchCodec dependency.

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [AMD GPU Architecture](https://github.com/ROCm/ROCm/blob/master/README.md#supported-gpus)
