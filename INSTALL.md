# Installation Guide

## System Requirements

### Hardware
- AMD GPU: RX 7700S (Navi 33) or similar
- VRAM: 8GB minimum
- RAM: 16GB+ recommended
- Disk: 50GB+ free space

### Software
- OS: Ubuntu 22.04+ (You have Ubuntu 25.10)
- Kernel: 5.15+ (You have 6.17)
- Python: 3.10+ (You have 3.13.7)

---

## Step 1: Install ROCm 6.2

ROCm (Radeon Open Compute) enables GPU acceleration for PyTorch.

### 1.1 Add ROCm Repository

```bash
# Download ROCm installer
cd /tmp
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.2.60200-1_all.deb

# Install the package
sudo apt install ./amdgpu-install_6.2.60200-1_all.deb
```

### 1.2 Install ROCm

```bash
# Install ROCm runtime and development tools
sudo amdgpu-install --usecase=rocm,graphics --no-dkms

# Add user to render and video groups
sudo usermod -aG render,video $USER

# Logout and login for group changes to take effect
```

### 1.3 Verify Installation

```bash
# Check ROCm version
rocminfo | head -20

# Check GPU detection
/opt/rocm/bin/rocm-smi

# You should see your RX 7700S listed
```

### 1.4 Environment Variables (Optional)

Add to `~/.bashrc`:

```bash
# ROCm paths
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocblas/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

# For RX 7700S (Navi 33) - may be needed for ROCm 6.2
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Apply:

```bash
source ~/.bashrc
```

---

## Step 2: Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    wget \
    curl
```

---

## Step 3: Create Python Virtual Environment

```bash
# Navigate to project directory
cd /media/ekagra/08B6457DB6456C6E/Audio/audio_upscaler

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## Step 4: Install PyTorch with ROCm

### Option A: Pre-built Wheels (Recommended)

```bash
# PyTorch with ROCm 6.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

### Option B: Nightly Build (If stable doesn't work)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

### Verify PyTorch ROCm

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Returns True for ROCm
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 6: Verify Setup

Run the verification script:

```bash
python -c "
import torch
import torchaudio
import librosa
print('All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'ROCm GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not detected\"}')
"
```

---

## Troubleshooting

### Issue: GPU Not Detected

```bash
# Check if kernel modules are loaded
lsmod | grep amdgpu

# Check permissions
ls -la /dev/kfd /dev/dri/

# Force override for Navi 33
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Issue: HIP Error

```bash
# Check HIP installation
/opt/rocm/bin/hipconfig

# Reinstall if needed
sudo apt install --reinstall rocm-hip-runtime
```

### Issue: Out of Memory During Training

Reduce batch size in `config.yaml`:

```yaml
batch_size: 4  # Reduce from 8
gradient_checkpointing: true
```

### Issue: Python 3.13 Compatibility

Some packages may not support Python 3.13. If issues occur:

```bash
# Install Python 3.12 via deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv

# Create venv with Python 3.12
python3.12 -m venv venv
```

---

## Post-Installation

1. Reboot your system:
   ```bash
   sudo reboot
   ```

2. After reboot, verify everything works:
   ```bash
   cd /media/ekagra/08B6457DB6456C6E/Audio/audio_upscaler
   source venv/bin/activate
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Proceed to data download:
   ```bash
   python download_data.py --size medium
   ```

---

## ROCm Version Compatibility

| PyTorch Version | ROCm Version | Python |
|-----------------|--------------|--------|
| 2.4.x           | 6.1          | 3.8-3.12 |
| 2.5.x           | 6.2          | 3.8-3.12 |
| 2.6.x (nightly) | 6.2          | 3.8-3.13 |

For Python 3.13, use PyTorch nightly or wait for stable 2.6+.

---

## Next Steps

After successful installation:

1. [DATA.md](DATA.md) - Download and prepare training data
2. [TRAINING.md](TRAINING.md) - Training guide
3. [INFERENCE.md](INFERENCE.md) - Using the model
