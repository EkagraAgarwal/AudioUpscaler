"""
Dataset for audio super-resolution training.
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T


class AudioUpscaleDataset(Dataset):
    """
    Dataset for audio super-resolution.

    Loads audio files and creates pairs of (compressed, original) for training.
    Supports memory-mapped WAV files for faster loading.
    """

    def __init__(
        self,
        audio_dir: str,
        audio_length: int = 262144,
        sample_rate: int = 44100,
        bitrate: int = 128,
        augment: bool = True,
        cache_compressed: bool = False,
        use_memmap: bool = False,
        wav_dir: Optional[str] = None,
        dynamic_bitrate: bool = False
    ):
        """
        Args:
            audio_dir: Directory containing audio files
            audio_length: Length of audio segment in samples
            sample_rate: Target sample rate
            bitrate: Target bitrate for compression (128, 192, 320) - used when dynamic_bitrate=False
            augment: Whether to apply augmentation
            cache_compressed: Whether to cache compressed versions
            use_memmap: Use memory-mapped WAV files for faster loading
            wav_dir: Directory containing WAV files (if use_memmap=True)
            dynamic_bitrate: If True, randomly select bitrate from [32, 48, 64, 96, 128] favoring lower values
        """
        self.audio_dir = Path(audio_dir)
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.dynamic_bitrate = dynamic_bitrate
        self.bitrate_options = [32, 48, 64, 96, 128]
        self.bitrate_weights = [0.30, 0.25, 0.25, 0.10, 0.10]
        self.augment = augment
        self.cache_compressed = cache_compressed
        self.use_memmap = use_memmap
        self.wav_dir = Path(wav_dir) if wav_dir else None

        self.audio_files = self._find_audio_files()

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        mode_str = "memory-mapped" if self.use_memmap else "on-demand"
        print(f"Found {len(self.audio_files)} audio files ({mode_str} loading)")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in directory."""
        if self.use_memmap and self.wav_dir:
            # Use WAV files from wav_dir
            extensions = {'.wav'}
            audio_files = []
            for ext in extensions:
                audio_files.extend(self.wav_dir.rglob(f"*{ext}"))
            return sorted(audio_files)
        else:
            # Use original audio files
            extensions = {'.mp3', '.flac', '.wav', '.ogg', '.aac'}
            audio_files = []
            for ext in extensions:
                audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
            return sorted(audio_files)

    def _load_audio_memmap(self, path: Path) -> torch.Tensor:
        """Load audio using memory mapping (fast)."""
        try:
            import soundfile as sf
            # Read WAV file directly into numpy array
            data, sr = sf.read(str(path), dtype='float32')
            
            # Verify sample rate
            if sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: {sr} != {self.sample_rate}")
            
            # Convert to tensor
            waveform = torch.from_numpy(data)
            
            # Handle stereo (should be mono after conversion)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=1)
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {path} with memmap: {e}")
            return None

    def _load_audio_legacy(self, path: Path) -> torch.Tensor:
        """Load audio using pydub (slower but supports more formats)."""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(str(path))

            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            if audio.channels > 1:
                audio = audio.set_channels(1)

            samples = audio.get_array_of_samples()
            waveform = torch.tensor(samples, dtype=torch.float32) / (2**15)

            return waveform

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio file."""
        # Choose loading method
        if self.use_memmap:
            waveform = self._load_audio_memmap(path)
        else:
            waveform = self._load_audio_legacy(path)
        
        if waveform is None:
            return None

        if len(waveform) < self.audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - len(waveform)))

        return waveform
    
    def _get_random_bitrate(self) -> int:
        """Get random bitrate favoring lower values (32, 48, 64)."""
        return random.choices(self.bitrate_options, weights=self.bitrate_weights, k=1)[0]

    def _compress_audio(self, waveform: torch.Tensor, bitrate: Optional[int] = None) -> torch.Tensor:
        """Simulate compression by applying low-pass filter and adding noise."""
        if bitrate is None:
            bitrate = self.bitrate
        freq_cutoff = min(bitrate * 100, self.sample_rate // 4)
        
        waveform_np = waveform.numpy()
        
        from scipy import signal
        nyquist = self.sample_rate // 2
        normalized_cutoff = min(freq_cutoff / nyquist, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        compressed = signal.filtfilt(b, a, waveform_np)
        
        noise_level = max(0.001, 0.1 - bitrate / 3200)
        noise = np.random.randn(len(compressed)) * noise_level
        compressed = compressed + noise
        
        compressed = np.clip(compressed, -1, 1)
        
        return torch.from_numpy(compressed).float()
    
    def _get_random_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Get random crop of audio."""
        if len(waveform) <= self.audio_length:
            return waveform
        
        start = random.randint(0, len(waveform) - self.audio_length)
        return waveform[start:start + self.audio_length]
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (compressed, original) pair."""
        audio_path = self.audio_files[idx]
        
        waveform = self._load_audio(audio_path)
        
        if waveform is None:
            waveform = torch.zeros(self.audio_length)
        
        if self.augment:
            waveform = self._get_random_crop(waveform)
        else:
            if len(waveform) > self.audio_length:
                waveform = waveform[:self.audio_length]
            elif len(waveform) < self.audio_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - len(waveform)))
        
        current_bitrate = self._get_random_bitrate() if self.dynamic_bitrate else self.bitrate
        compressed = self._compress_audio(waveform, bitrate=current_bitrate)
        
        if self.augment:
            gain = random.uniform(0.8, 1.2)
            compressed = compressed * gain
            waveform = waveform * gain
        
        return compressed, waveform


class AudioUpscaleDatasetPaired(Dataset):
    """
    Dataset that uses pre-compressed audio pairs.
    """
    
    def __init__(
        self,
        original_dir: str,
        compressed_dir: str,
        audio_length: int = 262144,
        sample_rate: int = 44100
    ):
        self.original_dir = Path(original_dir)
        self.compressed_dir = Path(compressed_dir)
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        
        self.audio_files = self._find_paired_files()
        print(f"Found {len(self.audio_files)} paired audio files")
    
    def _find_paired_files(self) -> List[Tuple[Path, Path]]:
        """Find matching pairs of original and compressed files."""
        extensions = {'.mp3', '.flac', '.wav'}
        pairs = []
        
        for ext in extensions:
            for orig_file in self.original_dir.rglob(f"*{ext}"):
                comp_file = self.compressed_dir / orig_file.relative_to(self.original_dir)
                if comp_file.exists():
                    pairs.append((orig_file, comp_file))
        
        return pairs
    
    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio file."""
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(str(path))
            
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            samples = audio.get_array_of_samples()
            waveform = torch.tensor(samples, dtype=torch.float32) / (2**15)
            return waveform
        except:
            return torch.zeros(self.audio_length)
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_path, comp_path = self.audio_files[idx]
        
        original = self._load_audio(orig_path)
        compressed = self._load_audio(comp_path)
        
        min_len = min(len(original), len(compressed), self.audio_length)
        original = original[:min_len]
        compressed = compressed[:min_len]
        
        if len(original) < self.audio_length:
            original = torch.nn.functional.pad(original, (0, self.audio_length - len(original)))
            compressed = torch.nn.functional.pad(compressed, (0, self.audio_length - len(compressed)))
        
        return compressed, original


def create_dataloaders(
    audio_dir: str,
    batch_size: int = 8,
    audio_length: int = 262144,
    sample_rate: int = 44100,
    bitrate: int = 128,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    use_memmap: bool = False,
    wav_dir: Optional[str] = None,
    dynamic_bitrate: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    dataset = AudioUpscaleDataset(
        audio_dir=audio_dir,
        audio_length=audio_length,
        sample_rate=sample_rate,
        bitrate=bitrate,
        augment=True,
        use_memmap=use_memmap,
        wav_dir=wav_dir,
        dynamic_bitrate=dynamic_bitrate
    )

    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = AudioUpscaleDataset(
        audio_dir="data/raw/fma_small",
        audio_length=65536,
        sample_rate=44100,
        bitrate=128
    )
    
    compressed, original = dataset[0]
    print(f"Compressed shape: {compressed.shape}")
    print(f"Original shape: {original.shape}")
