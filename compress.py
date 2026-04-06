#!/usr/bin/env python3
"""
Create compressed audio pairs for training.
Converts high-quality audio to lower bitrates (MP3, AAC) for training pairs.

Usage:
    python compress.py --input data/raw --output data/compressed
    python compress.py --input data/raw --output data/compressed --bitrates 128,192,320
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import shutil

# Audio formats and extensions
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.wav', '.ogg', '.aac', '.m4a'}

# Default compression settings
DEFAULT_BITRATES = {
    'mp3': [128, 192, 320],
    'aac': [96, 128]
}

def get_audio_files(input_dir: Path) -> list:
    """Get all audio files in directory."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(audio_files)

def compress_audio(
    input_file: Path,
    output_file: Path,
    bitrate: int,
    codec: str = 'mp3',
    sample_rate: int = 44100,
    channels: int = 2
) -> bool:
    """
    Compress audio file using ffmpeg.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output compressed file
        bitrate: Target bitrate in kbps
        codec: Audio codec ('mp3' or 'aac')
        sample_rate: Target sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
    
    Returns:
        True if successful, False otherwise
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    codec_map = {
        'mp3': 'libmp3lame',
        'aac': 'aac'
    }
    
    ffmpeg_codec = codec_map.get(codec, 'libmp3lame')
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_file),
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-c:a', ffmpeg_codec,
        '-b:a', f'{bitrate}k',
        '-loglevel', 'error',
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {input_file}: {e}")
        return False

def process_file(args):
    """Process a single audio file (for multiprocessing)."""
    input_file, output_base, bitrates, codec, sample_rate, channels = args
    results = []
    
    for bitrate in bitrates:
        output_file = output_base / f"{bitrate}kbps" / input_file.name
        output_file = output_file.with_suffix('.mp3') if codec == 'mp3' else output_file.with_suffix('.m4a')
        
        success = compress_audio(
            input_file,
            output_file,
            bitrate,
            codec,
            sample_rate,
            channels
        )
        
        results.append((input_file, bitrate, success))
    
    return results

def split_dataset(
    audio_files: list,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> dict:
    """Split audio files into train/val/test sets."""
    import random
    random.seed(seed)
    random.shuffle(audio_files)
    
    n_total = len(audio_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    splits = {
        'train': audio_files[:n_train],
        'val': audio_files[n_train:n_train + n_val],
        'test': audio_files[n_train + n_val:]
    }
    
    return splits

def main():
    parser = argparse.ArgumentParser(
        description="Create compressed audio pairs for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compress.py --input data/raw/fma_medium --output data/compressed
    python compress.py --input data/raw --output data/compressed --bitrates 128,192
    python compress.py --input data/raw --output data/compressed --codec aac --bitrates 96,128
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing audio files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/compressed',
        help='Output directory for compressed files (default: data/compressed)'
    )
    
    parser.add_argument(
        '--bitrates',
        type=str,
        default='128,192',
        help='Comma-separated bitrates in kbps (default: 128,192)'
    )
    
    parser.add_argument(
        '--codec',
        type=str,
        choices=['mp3', 'aac', 'both'],
        default='mp3',
        help='Audio codec to use (default: mp3)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Target sample rate in Hz (default: 44100)'
    )
    
    parser.add_argument(
        '--channels',
        type=int,
        default=2,
        choices=[1, 2],
        help='Number of audio channels (default: 2 for stereo)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without processing'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    bitrates = [int(b) for b in args.bitrates.split(',')]
    
    print(f"\n{'='*60}")
    print("Audio Compression Pipeline")
    print(f"{'='*60}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Codec:   {args.codec.upper()}")
    print(f"Bitrates: {bitrates} kbps")
    print(f"Sample Rate: {args.sample_rate} Hz")
    print(f"Channels: {'Stereo' if args.channels == 2 else 'Mono'}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")
    
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        print(f"Error: No audio files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(audio_files):,} audio files")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process these files:")
        for i, f in enumerate(audio_files[:5]):
            print(f"  {i+1}. {f.name}")
        print(f"  ... and {len(audio_files) - 5} more")
        print(f"\n[DRY RUN] Would create compressed versions:")
        for br in bitrates:
            print(f"  {output_dir}/mp3_{br}kbps/")
        sys.exit(0)
    
    splits = split_dataset(
        audio_files,
        args.train_split,
        args.val_split,
        args.seed
    )
    
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} set ({len(files):,} files)...")
        
        split_output = output_dir / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        
        codecs_to_process = ['mp3', 'aac'] if args.codec == 'both' else [args.codec]
        
        for codec in codecs_to_process:
            for bitrate in bitrates:
                bitrate_dir = split_output / f"{codec}_{bitrate}kbps"
                bitrate_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for split_name, files in splits.items():
        print(f"\nCompressing {split_name} ({len(files):,} files)...")
        
        for audio_file in tqdm(files, desc=f"{split_name}"):
            for codec in (['mp3', 'aac'] if args.codec == 'both' else [args.codec]):
                for bitrate in bitrates:
                    rel_path = audio_file.relative_to(input_dir)
                    output_base = output_dir / split_name
                    
                    output_file = (
                        output_base / f"{codec}_{bitrate}kbps" / rel_path.name
                    )
                    output_file = output_file.with_suffix(
                        '.mp3' if codec == 'mp3' else '.m4a'
                    )
                    
                    if output_file.exists():
                        success_count += 1
                        continue
                    
                    success = compress_audio(
                        audio_file,
                        output_file,
                        bitrate,
                        codec,
                        args.sample_rate,
                        args.channels
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
    
    print(f"\n{'='*60}")
    print("Compression Complete")
    print(f"{'='*60}")
    print(f"Successfully compressed: {success_count:,}")
    print(f"Errors: {error_count:,}")
    
    disk_usage = subprocess.run(
        ['du', '-sh', str(output_dir)],
        capture_output=True,
        text=True
    )
    if disk_usage.returncode == 0:
        print(f"Output size: {disk_usage.stdout.strip().split()[0]}")
    
    print(f"\nOutput structure:")
    for split_name in splits.keys():
        print(f"  {split_name}/")
        for codec in (['mp3', 'aac'] if args.codec == 'both' else [args.codec]):
            for bitrate in bitrates:
                print(f"    {codec}_{bitrate}kbps/")
    
    print(f"\n✓ Compression complete!")
    print(f"\nNext step: python src/train.py")

if __name__ == "__main__":
    main()
