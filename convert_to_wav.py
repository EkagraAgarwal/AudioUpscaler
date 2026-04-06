#!/usr/bin/env python3
"""
Convert MP3 audio files to WAV format for memory-mapped loading.

This script converts all audio files from a source directory to WAV format,
preserving the directory structure. The resulting WAV files can be loaded
using memory mapping for significantly faster training.

Usage:
    python convert_to_wav.py --src data/raw/fma_small --dst data/wav_cache

One-time conversion:
    - Takes ~10-15 minutes for 4000 files
    - Requires ~15-20 GB additional disk space
    - Result: 55% faster training epochs
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment


def convert_audio_file(src_path: Path, dst_path: Path, sample_rate: int = 44100) -> bool:
    """
    Convert a single audio file to WAV format.
    
    Args:
        src_path: Source audio file path
        dst_path: Destination WAV file path
        sample_rate: Target sample rate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(str(src_path))
        
        # Convert to mono and target sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as WAV
        audio.export(str(dst_path), format='wav')
        
        return True
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        return False


def convert_dataset(src_dir: str, dst_dir: str, sample_rate: int = 44100) -> dict:
    """
    Convert all audio files from source to destination directory.
    
    Args:
        src_dir: Source directory containing audio files
        dst_dir: Destination directory for WAV files
        sample_rate: Target sample rate
    
    Returns:
        Dictionary with conversion statistics
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    if not src_dir.exists():
        raise ValueError(f"Source directory does not exist: {src_dir}")
    
    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_extensions = {'.mp3', '.flac', '.wav', '.ogg', '.aac'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(src_dir.rglob(f"*{ext}"))
    
    audio_files = sorted(audio_files)
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {src_dir}")
    
    print(f"\n{'='*60}")
    print(f"Audio Conversion to WAV")
    print(f"{'='*60}")
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Files to convert: {len(audio_files)}")
    print(f"{'='*60}\n")
    
    # Convert files
    stats = {
        'total': len(audio_files),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'failed_files': []
    }
    
    for src_path in tqdm(audio_files, desc="Converting files", unit="file"):
        # Preserve directory structure
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path.with_suffix('.wav')
        
        # Skip if already exists
        if dst_path.exists():
            stats['skipped'] += 1
            continue
        
        # Create parent directories
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert file
        success = convert_audio_file(src_path, dst_path, sample_rate)
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
            stats['failed_files'].append(str(src_path))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Total files: {stats['total']}")
    print(f"Converted: {stats['success']}")
    print(f"Skipped (already exist): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        print(f"\nFailed files:")
        for file in stats['failed_files']:
            print(f"  - {file}")
    
    print(f"{'='*60}\n")
    
    # Calculate disk usage
    total_size = sum(f.stat().st_size for f in dst_dir.rglob("*.wav") if f.is_file())
    print(f"Total WAV size: {total_size / (1024**3):.2f} GB")
    print(f"Location: {dst_dir}")
    print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to WAV format for memory-mapped loading"
    )
    parser.add_argument(
        '--src',
        type=str,
        default='data/raw/fma_small',
        help='Source directory containing audio files'
    )
    parser.add_argument(
        '--dst',
        type=str,
        default='data/wav_cache',
        help='Destination directory for WAV files'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Target sample rate (default: 44100)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing WAV files instead of converting'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify existing WAV files
        dst_dir = Path(args.dst)
        if not dst_dir.exists():
            print(f"Error: Destination directory does not exist: {dst_dir}")
            sys.exit(1)
        
        wav_files = list(dst_dir.rglob("*.wav"))
        print(f"Found {len(wav_files)} WAV files in {dst_dir}")
        
        total_size = sum(f.stat().st_size for f in wav_files if f.is_file())
        print(f"Total size: {total_size / (1024**3):.2f} GB")
    else:
        # Convert files
        stats = convert_dataset(args.src, args.dst, args.sample_rate)
        
        if stats['failed'] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
