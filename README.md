# Audio Upscaler

A deep learning model that upscales low bitrate/lossy compressed audio to higher bitrate or near-lossless quality using PyTorch with AMD ROCm acceleration.

## Overview

This project implements a Waveform U-Net architecture to enhance audio quality by learning to reconstruct high-fidelity audio from compressed versions. The model is trained on paired data: high-quality lossless audio and artificially compressed versions.

## Features

- **ROCm Acceleration**: Optimized for AMD GPUs (tested on RX 7700S)
- **Waveform U-Net**: Direct 1D convolution on audio waveforms
- **Multi-Resolution STFT Loss**: Captures spectral details at multiple scales
- **Mixed Precision Training**: Faster training with FP16
- **FLAC/WAV Output**: Export upscaled audio in lossless formats

## Project Structure

```
audio_upscaler/
├── data/
│   ├── raw/              # Downloaded FLAC/WAV files
│   ├── compressed/       # Artificially compressed versions
│   └── checkpoints/      # Model weights
├── src/
│   ├── model.py          # U-Net 1D architecture
│   ├── dataset.py        # Data loading and preprocessing
│   ├── train.py          # Training loop
│   ├── inference.py      # Upscaling script
│   └── utils.py          # Audio processing utilities
├── download_data.py      # FMA download script
├── compress.py           # Create training pairs
├── config.yaml           # Hyperparameters
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Requirements

### Hardware
- AMD GPU with ROCm support (RX 7000 series recommended)
- 8GB+ VRAM
- 50GB+ disk space for data

### Software
- Ubuntu 22.04+
- ROCm 6.2+
- Python 3.10+
- FFmpeg

## Installation

### 1. Install ROCm

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.2.x_all.deb
sudo apt install ./amdgpu-install_6.2.x_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm,graphics
sudo usermod -aG render,video $USER

# Reboot required
sudo reboot
```

### 2. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install FFmpeg

```bash
sudo apt install ffmpeg
```

## Quick Start

### 1. Download Dataset

```bash
python download_data.py --size medium --output data/raw/
```

### 2. Create Compressed Pairs

```bash
python compress.py --input data/raw/ --output data/compressed/
```

### 3. Train Model

```bash
python src/train.py --config config.yaml
```

### 4. Upscale Audio

```bash
python src/inference.py --input low_quality.mp3 --output upscaled.flac --checkpoint data/checkpoints/best_model.pt
```

## Model Architecture

### U-Net 1D for Audio

```
Input: (batch, 1, 262144)  # ~6 seconds at 44.1kHz
    │
    ├── Encoder (Contracting Path)
    │   ├── Block 1: Conv1D(64) + Downsample
    │   ├── Block 2: Conv1D(128) + Downsample
    │   ├── Block 3: Conv1D(256) + Downsample
    │   ├── Block 4: Conv1D(512) + Downsample
    │   └── Block 5: Conv1D(512) + Downsample
    │
    ├── Bottleneck: Conv1D(1024)
    │
    └── Decoder (Expanding Path)
        ├── Block 1: Upsample + Conv1D(512) + Skip Connection
        ├── Block 2: Upsample + Conv1D(256) + Skip Connection
        ├── Block 3: Upsample + Conv1D(128) + Skip Connection
        ├── Block 4: Upsample + Conv1D(64) + Skip Connection
        └── Block 5: Upsample + Conv1D(32) + Skip Connection
    │
Output: (batch, 1, 262144)  # Reconstructed waveform
```

## Loss Functions

1. **L1 Waveform Loss**: Direct reconstruction error
2. **Multi-Resolution STFT Loss**: Spectral accuracy at multiple scales
3. **Perceptual Loss** (optional): Feature matching from pretrained encoder

## Training Configuration

Default hyperparameters in `config.yaml`:

```yaml
batch_size: 8
learning_rate: 3e-4
epochs: 50
audio_length: 262144
sample_rate: 44100
mixed_precision: true
```

## Monitoring Training

```bash
tensorboard --logdir runs/
```

## Evaluation Metrics

- Signal-to-Noise Ratio (SNR)
- Multi-resolution STFT distance
- Subjective A/B listening tests

## Troubleshooting

### ROCm Issues

```bash
# Check ROCm installation
rocminfo

# Check GPU visibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7700S
```

### Out of Memory

Reduce batch size or enable gradient checkpointing in config.yaml

## References

- [Free Music Archive (FMA)](https://github.com/mdeff/fma)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646)

## License

This project is for educational and research purposes. Training data licensing follows FMA's Creative Commons licenses.

## Author

Audio Upscaler Project - 2026
