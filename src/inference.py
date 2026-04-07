#!/usr/bin/env python3
"""
Inference script for audio super-resolution.
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from scipy import signal

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import AudioUNet1D, AudioUNet1DSimple


def compress_to_32kbps(waveform, sample_rate=44100):
    """Simulate 32kbps compression with severe low-pass filtering."""
    freq_cutoff = min(3200, sample_rate // 4)
    
    waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
    
    nyquist = sample_rate // 2
    normalized_cutoff = min(freq_cutoff / nyquist, 0.99)
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    compressed = signal.filtfilt(b, a, waveform_np)
    
    noise_level = 0.08
    noise = np.random.randn(len(compressed)) * noise_level
    compressed = compressed + noise
    compressed = np.clip(compressed, -1, 1)
    
    return torch.from_numpy(compressed).float()


def upscale_audio(
    model,
    waveform,
    device,
    chunk_size=32768,
    overlap=0.5
):
    """Upscale audio using the model with overlap-add for long files."""
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    total_samples = waveform.shape[-1]
    hop_size = int(chunk_size * (1 - overlap))
    
    output = torch.zeros(1, total_samples, device=device)
    window = torch.hann_window(chunk_size, device=device)
    
    with torch.no_grad():
        start = 0
        while start < total_samples:
            end = min(start + chunk_size, total_samples)
            chunk = waveform[:, start:end]
            
            if chunk.shape[-1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[-1]))
            
            chunk = chunk.to(device)
            upscaled = model(chunk)
            
            if isinstance(upscaled, tuple):
                upscaled = upscaled[0]
            
            window_len = end - start
            
            if start == 0:
                output[:, start:end] = upscaled[:, :window_len]
            else:
                output[:, start:end] = output[:, start:end] * (1 - window[:window_len]) + upscaled[:, :window_len] * window[:window_len]
            
            start += hop_size
    
    return output.cpu()


def main():
    parser = argparse.ArgumentParser(description="Upscale audio using trained model")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output audio file")
    parser.add_argument("--compressed-output", type=str, default=None, help="Output for 32kbps compressed version")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--chunk-size", type=int, default=32768, help="Chunk size for processing")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = AudioUNet1D(in_channels=1, base_channels=48, depth=4, use_dilated_bottleneck=True, use_interpolation_upsampling=True)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    print(f"\nLoading audio from {args.input}...")
    import soundfile as sf
    waveform_np, sample_rate = sf.read(args.input)
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=1)
    waveform = waveform.unsqueeze(0)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"Audio: {waveform.shape[-1]} samples at {sample_rate}Hz")
    
    print("\nCreating 32kbps compressed version...")
    compressed = compress_to_32kbps(waveform.squeeze(0), sample_rate)
    
    input_path = Path(args.input)
    
    compressed_path = args.compressed_output
    if compressed_path is None:
        compressed_path = str(input_path.parent / f"{input_path.stem}_32kbps.wav")
    else:
        os.makedirs(os.path.dirname(compressed_path) if os.path.dirname(compressed_path) else '.', exist_ok=True)
    import soundfile as sf
    sf.write(compressed_path, compressed.unsqueeze(0).numpy().T, sample_rate)
    print(f"Saved 32kbps version to: {compressed_path}")
    
    print("\nUpscaling audio...")
    upscaled = upscale_audio(model, compressed.unsqueeze(0), device, chunk_size=args.chunk_size)
    
    output_path = args.output
    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_upscaled.wav")
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    import soundfile as sf
    sf.write(output_path, upscaled.squeeze(0).numpy().T, sample_rate)
    print(f"Saved upscaled version to: {output_path}")
    
    print(f"\nFiles ready for comparison:")
    print(f"  Original:  {args.input}")
    print(f"  32kbps:    {compressed_path}")
    print(f"  Upscaled:  {output_path}")


if __name__ == "__main__":
    main()
