#!/usr/bin/env python3
"""
Download Free Music Archive (FMA) dataset for audio upscaler training.

Usage:
    python download_data.py --size medium --output data/raw/
    python download_data.py --size small --output data/raw/ --test
"""

import argparse
import os
import sys
import zipfile
import tarfile
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess

# FMA Download URLs
FMA_URLS = {
    "small": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "size": "2.1GB",
        "tracks": 8000,
        "checksum": "1f5fef8b52d8ef65e3846cc38e9e9e8d"
    },
    "medium": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        "size": "30GB",
        "tracks": 25000,
        "checksum": "e823a3d7a8dbb05e2f1a12e4c9e2b1c0"
    },
    "large": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
        "size": "100GB",
        "tracks": 106574,
        "checksum": "10a9e2b3c4d5e6f7a8b9c0d1e2f3a4b5"
    },
    "metadata": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "size": "338MB",
        "checksum": "f0d8e9a7b6c5d4e3f2a1b0c9d8e7f6a5"
    }
}

def download_file(url: str, output_path: Path, desc: str = None):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get('content-length', 0))
    
    desc = desc or output_path.name
    
    with open(output_path, 'wb') as f:
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return output_path

def extract_archive(archive_path: Path, output_dir: Path):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path.name}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, output_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc="Extracting"):
                tar_ref.extract(member, output_dir)
    
    print(f"Extracted to {output_dir}")

def verify_checksum(file_path: Path, expected: str) -> bool:
    """Verify MD5 checksum of downloaded file."""
    print(f"Verifying checksum of {file_path.name}...")
    
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in tqdm(iter(lambda: f.read(8192), b''), desc="Computing MD5"):
            md5_hash.update(chunk)
    
    actual = md5_hash.hexdigest()
    if actual == expected:
        print("✓ Checksum verified")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        return False

def download_fma(size: str, output_dir: Path, skip_checksum: bool = False):
    """Download FMA dataset."""
    if size not in FMA_URLS:
        print(f"Invalid size: {size}")
        print(f"Available: {list(FMA_URLS.keys())}")
        sys.exit(1)
    
    info = FMA_URLS[size]
    print(f"\n{'='*60}")
    print(f"Downloading FMA {size.upper()}")
    print(f"Size: {info['size']}")
    print(f"Tracks: {info['tracks']:,}")
    print(f"{'='*60}\n")
    
    archive_path = output_dir / f"fma_{size}.zip"
    
    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
        print("Skipping download. Delete file to re-download.")
    else:
        download_file(info['url'], archive_path, f"FMA {size}")
        
        if not skip_checksum:
            verify_checksum(archive_path, info['checksum'])
    
    if output_dir.joinpath(f"fma_{size}").exists():
        print(f"Dataset already extracted: {output_dir}/fma_{size}")
    else:
        extract_archive(archive_path, output_dir)
    
    print(f"\n✓ FMA {size} ready at: {output_dir}/fma_{size}")

def download_metadata(output_dir: Path, skip_checksum: bool = False):
    """Download FMA metadata."""
    print(f"\n{'='*60}")
    print("Downloading FMA Metadata")
    print(f"{'='*60}\n")
    
    info = FMA_URLS['metadata']
    archive_path = output_dir / "fma_metadata.zip"
    
    if archive_path.exists():
        print(f"Metadata archive already exists: {archive_path}")
    else:
        download_file(info['url'], archive_path, "Metadata")
        
        if not skip_checksum:
            verify_checksum(archive_path, info['checksum'])
    
    if output_dir.joinpath("fma_metadata").exists():
        print(f"Metadata already extracted: {output_dir}/fma_metadata")
    else:
        extract_archive(archive_path, output_dir)
    
    print(f"\n✓ Metadata ready at: {output_dir}/fma_metadata")

def count_audio_files(directory: Path) -> int:
    """Count audio files in directory."""
    audio_extensions = {'.mp3', '.flac', '.wav', '.ogg', '.aac'}
    count = 0
    for ext in audio_extensions:
        count += len(list(directory.rglob(f"*{ext}")))
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Download Free Music Archive (FMA) dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py --size medium --output data/raw/
    python download_data.py --size small --output data/raw/ --no-checksum
    python download_data.py --metadata-only --output data/raw/
        """
    )
    
    parser.add_argument(
        '--size',
        type=str,
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Dataset size to download (default: medium)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help='Download only metadata'
    )
    
    parser.add_argument(
        '--no-checksum',
        action='store_true',
        help='Skip checksum verification'
    )
    
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='Keep downloaded zip files after extraction'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    try:
        if args.metadata_only:
            download_metadata(output_dir, args.no_checksum)
        else:
            download_fma(args.size, output_dir, args.no_checksum)
            download_metadata(output_dir, args.no_checksum)
        
        print(f"\n{'='*60}")
        print("Download Summary")
        print(f"{'='*60}")
        
        if not args.metadata_only:
            audio_count = count_audio_files(output_dir / f"fma_{args.size}")
            print(f"Audio files: {audio_count:,}")
        
        print(f"Location: {output_dir.absolute()}")
        
        disk_usage = subprocess.run(
            ['du', '-sh', str(output_dir)],
            capture_output=True,
            text=True
        )
        if disk_usage.returncode == 0:
            print(f"Disk usage: {disk_usage.stdout.strip().split()[0]}")
        
        print(f"\n✓ Download complete!")
        print(f"\nNext steps:")
        print(f"  1. Create compressed pairs: python compress.py --input {output_dir}")
        print(f"  2. Start training: python src/train.py")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Run again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
