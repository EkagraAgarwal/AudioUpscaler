# Data Guide

## Overview

This project uses the Free Music Archive (FMA) dataset for training. The FMA is a high-quality, Creative Commons-licensed collection suitable for music information retrieval and audio machine learning.

## Data Sources

### Primary: Free Music Archive (FMA)

- **FMA Medium**: ~30GB, 25,000 songs (recommended for prototype)
- **FMA Large**: ~100GB, 106,000 songs (for full training)
- **FMA Small**: ~2GB, 8,000 songs (for testing)

Download from: https://github.com/mdeff/fma

### Alternative Sources

- **Jamendo**: https://www.jamendo.com (free music for ML)
- **Internet Archive**: Live music collections
- **YouTube**: Via yt-dlp (for additional data)

---

## Data Pipeline

```
1. Download original FLAC/WAV files
   └─> data/raw/

2. Create compressed versions
   ├─> MP3 128kbps ─> data/compressed/mp3_128/
   ├─> MP3 192kbps ─> data/compressed/mp3_192/
   └─> AAC 96kbps  ─> data/compressed/aac_96/

3. Split into train/val/test
   └─> 80% train, 10% val, 10% test
```

---

## Downloading FMA

### Method 1: Using the Download Script

```bash
# Activate environment
source venv/bin/activate

# Download FMA medium (~30GB)
python download_data.py --size medium --output data/raw/

# Or download small for testing (~2GB)
python download_data.py --size small --output data/raw/
```

### Method 2: Manual Download

1. Visit: https://os.unil.cloud.switch.ch/fma/

2. Download required files:
   ```bash
   cd data/raw/
   
   # FMA Medium
   wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip
   wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
   
   # Extract
   unzip fma_medium.zip
   unzip fma_metadata.zip
   ```

3. Verify download:
   ```bash
   # Should see ~25,000 audio files
   find data/raw/fma_medium -name "*.mp3" | wc -l
   ```

---

## Data Structure After Download

```
data/
├── raw/
│   ├── fma_medium/
│   │   ├── 000/
│   │   │   ├── 000002.mp3
│   │   │   ├── 000005.mp3
│   │   │   └── ...
│   │   ├── 001/
│   │   └── ... (252 folders)
│   └── fma_metadata/
│       ├── tracks.csv
│       ├── genres.csv
│       └── ...
│
├── compressed/
│   ├── mp3_128/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── mp3_192/
│       ├── train/
│       ├── val/
│       └── test/
│
└── checkpoints/
    └── .gitkeep
```

---

## Creating Compressed Pairs

The training pairs consist of:
- **Input**: Low quality (compressed)
- **Target**: High quality (original)

### Run Compression Script

```bash
# Create all compression variants
python compress.py --input data/raw/fma_medium --output data/compressed

# Specific bitrate
python compress.py --input data/raw --output data/compressed --bitrates 128,192

# With specific sample rate
python compress.py --input data/raw --output data/compressed --sample-rate 44100
```

### Compression Settings

| Format | Bitrate | Quality |
|--------|---------|---------|
| MP3    | 128 kbps| Low (high compression artifacts) |
| MP3    | 192 kbps| Medium |
| MP3    | 320 kbps| High |
| AAC    | 96 kbps | Low (different artifacts) |

---

## Audio Format Requirements

### Target Format (for training)

All audio will be converted to:
- **Sample Rate**: 44.1 kHz
- **Channels**: Stereo (2 channels)
- **Bit Depth**: 16-bit (for FLAC/WAV output)
- **Duration**: Full tracks, split into ~6s chunks

### Supported Input Formats

- MP3
- FLAC
- WAV
- OGG
- AAC
- M4A

---

## Dataset Split

### Automatic Split (Recommended)

The dataset class handles splitting automatically:

```yaml
# config.yaml
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 42
```

### Manual Split

```bash
# Use metadata to split by genre
python scripts/split_dataset.py --metadata data/raw/fma_metadata/tracks.csv
```

---

## Data Augmentation

Applied during training:

1. **Random Crop**: Extract random ~6s segments
2. **Volume Perturbation**: ±6 dB
3. **Gain Normalization**: LUFS normalization

---

## Storage Requirements

| Component | Size |
|-----------|------|
| FMA Medium (original) | ~30 GB |
| Compressed versions (128 + 192) | ~3 GB |
| Processed tensors (cached) | ~10 GB |
| **Total** | **~45 GB** |

---

## Troubleshooting

### Download Fails

```bash
# Use curl with retry
curl -C - -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip

# Or use aria2c for parallel download
sudo apt install aria2
aria2c -x 16 https://os.unil.cloud.switch.ch/fma/fma_medium.zip
```

### Disk Space Issues

```bash
# Check available space
df -h /media/ekagra/08B6457DB6456C6E

# Clean up if needed
rm -rf data/raw/*.zip  # After extraction
```

### Corrupted Files

The dataset class automatically skips corrupted files. If you see errors:

```bash
# Validate all files
python scripts/validate_audio.py --dir data/raw/fma_medium
```

---

## Next Steps

1. Run download script:
   ```bash
   python download_data.py --size medium
   ```

2. Create compressed pairs:
   ```bash
   python compress.py --input data/raw --output data/compressed
   ```

3. Proceed to training:
   ```bash
   python src/train.py --config config.yaml
   ```

---

## Dataset License

FMA content is licensed under Creative Commons. Each track has specific licensing:
- CC BY
- CC BY-SA
- CC BY-NC
- CC BY-NC-SA
- CC BY-ND
- CC BY-NC-ND

Check `tracks.csv` for individual track licenses.
