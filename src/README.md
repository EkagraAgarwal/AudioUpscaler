# Audio Upscaler

A PyTorch-based audio super-resolution model that upscales low bitrate/lossy compressed audio to higher quality using AMD ROCm acceleration.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python download_data.py --size medium

# Train
python src/train.py

# Inference
python src/inference.py --input audio.mp3 --output upscaled.flac
```

## Documentation

- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [DATA.md](DATA.md) - Data download and preparation
- [README.md](README.md) - Full project documentation

## Status

🚧 **Project Setup Complete** - Ready for implementation
