#!/usr/bin/env python3
"""
Benchmark script to compare training optimizations.

Tests:
1. Baseline (no optimizations)
2. Mixed Precision (AMP)
3. torch.compile()
4. Optimized data loading
5. Combined optimizations
"""

import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, 'src')
from model import AudioUNet1D
from dataset import AudioUpscaleDataset
from utils import multi_resolution_stft_loss


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benchmark_baseline(
    model, 
    dataloader, 
    optimizer,
    device,
    num_batches=20,
    warmup_batches=5
):
    """Baseline training without optimizations."""
    model.train()
    
    # Warmup
    print(f"  Warming up ({warmup_batches} batches)...")
    for i, (compressed, original) in enumerate(dataloader):
        if i >= warmup_batches:
            break
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        output = model(compressed)
        l1_loss = nn.functional.l1_loss(output, original)
        stft_loss = multi_resolution_stft_loss(output, original)
        loss = l1_loss + stft_loss
        loss.backward()
        optimizer.step()
    
    # Benchmark
    print(f"  Benchmarking ({num_batches} batches)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    batch_times = []
    batch_idx = 0
    for compressed, original in dataloader:
        if batch_idx >= num_batches:
            break
            
        batch_start = time.time()
        
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        output = model(compressed)
        l1_loss = nn.functional.l1_loss(output, original)
        stft_loss = multi_resolution_stft_loss(output, original)
        loss = l1_loss + stft_loss
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        batch_end = time.time()
        
        batch_times.append(batch_end - batch_start)
        batch_idx += 1
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = num_batches / total_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput,
        'batch_times': batch_times
    }


def benchmark_amp(
    model, 
    dataloader, 
    optimizer,
    device,
    num_batches=20,
    warmup_batches=5
):
    """Training with Automatic Mixed Precision."""
    model.train()
    scaler = GradScaler()
    
    # Warmup
    print(f"  Warming up ({warmup_batches} batches)...")
    for i, (compressed, original) in enumerate(dataloader):
        if i >= warmup_batches:
            break
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = model(compressed)
            l1_loss = nn.functional.l1_loss(output, original)
            stft_loss = multi_resolution_stft_loss(output, original)
            loss = l1_loss + stft_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Benchmark
    print(f"  Benchmarking ({num_batches} batches)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    batch_times = []
    batch_idx = 0
    for compressed, original in dataloader:
        if batch_idx >= num_batches:
            break
            
        batch_start = time.time()
        
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = model(compressed)
            l1_loss = nn.functional.l1_loss(output, original)
            stft_loss = multi_resolution_stft_loss(output, original)
            loss = l1_loss + stft_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        batch_end = time.time()
        
        batch_times.append(batch_end - batch_start)
        batch_idx += 1
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = num_batches / total_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput,
        'batch_times': batch_times
    }


def benchmark_compile(
    model, 
    dataloader, 
    optimizer,
    device,
    num_batches=20,
    warmup_batches=5
):
    """Training with torch.compile()."""
    # Compile the model
    print("  Compiling model...")
    compiled_model = torch.compile(model)
    compiled_model.train()
    
    # Warmup (includes compilation time)
    print(f"  Warming up ({warmup_batches} batches)...")
    for i, (compressed, original) in enumerate(dataloader):
        if i >= warmup_batches:
            break
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        output = compiled_model(compressed)
        l1_loss = nn.functional.l1_loss(output, original)
        stft_loss = multi_resolution_stft_loss(output, original)
        loss = l1_loss + stft_loss
        loss.backward()
        optimizer.step()
    
    # Benchmark
    print(f"  Benchmarking ({num_batches} batches)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    batch_times = []
    batch_idx = 0
    for compressed, original in dataloader:
        if batch_idx >= num_batches:
            break
            
        batch_start = time.time()
        
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        output = compiled_model(compressed)
        l1_loss = nn.functional.l1_loss(output, original)
        stft_loss = multi_resolution_stft_loss(output, original)
        loss = l1_loss + stft_loss
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        batch_end = time.time()
        
        batch_times.append(batch_end - batch_start)
        batch_idx += 1
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = num_batches / total_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput,
        'batch_times': batch_times
    }


def benchmark_combined(
    model, 
    dataloader, 
    optimizer,
    device,
    num_batches=20,
    warmup_batches=5
):
    """Training with all optimizations combined."""
    # Compile the model
    print("  Compiling model...")
    compiled_model = torch.compile(model)
    compiled_model.train()
    scaler = GradScaler()
    
    # Warmup (includes compilation time)
    print(f"  Warming up ({warmup_batches} batches)...")
    for i, (compressed, original) in enumerate(dataloader):
        if i >= warmup_batches:
            break
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = compiled_model(compressed)
            l1_loss = nn.functional.l1_loss(output, original)
            stft_loss = multi_resolution_stft_loss(output, original)
            loss = l1_loss + stft_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Benchmark
    print(f"  Benchmarking ({num_batches} batches)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    batch_times = []
    batch_idx = 0
    for compressed, original in dataloader:
        if batch_idx >= num_batches:
            break
            
        batch_start = time.time()
        
        compressed = compressed.to(device)
        original = original.to(device)
        
        optimizer.zero_grad()
        with autocast():
            output = compiled_model(compressed)
            l1_loss = nn.functional.l1_loss(output, original)
            stft_loss = multi_resolution_stft_loss(output, original)
            loss = l1_loss + stft_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        batch_end = time.time()
        
        batch_times.append(batch_end - batch_start)
        batch_idx += 1
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = num_batches / total_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput,
        'batch_times': batch_times
    }


def print_results(name, results, baseline_results=None):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Avg batch time: {results['avg_batch_time']*1000:.1f}ms")
    print(f"  Throughput: {results['throughput']:.2f} batches/s")
    
    if baseline_results:
        speedup = baseline_results['avg_batch_time'] / results['avg_batch_time']
        improvement = (1 - results['avg_batch_time'] / baseline_results['avg_batch_time']) * 100
        print(f"  Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training optimizations")
    parser.add_argument('--audio-dir', type=str, default='data/raw/fma_small')
    parser.add_argument('--wav-dir', type=str, default='data/wav_cache')
    parser.add_argument('--use-memmap', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--audio-length', type=int, default=32768)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches to benchmark')
    parser.add_argument('--warmup-batches', type=int, default=5, help='Number of warmup batches')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataset
    print(f"\nLoading dataset...")
    dataset = AudioUpscaleDataset(
        audio_dir=args.audio_dir,
        wav_dir=args.wav_dir if args.use_memmap else None,
        use_memmap=args.use_memmap,
        audio_length=args.audio_length,
        sample_rate=44100
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create model
    print(f"\nCreating model...")
    model = AudioUNet1D(in_channels=1, base_channels=32, depth=4)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    print(f"\n{'='*60}")
    print(f"Running Benchmarks")
    print(f"{'='*60}")
    print(f"Batch size: {args.batch_size}")
    print(f"Audio length: {args.audio_length}")
    print(f"Batches per test: {args.num_batches}")
    print(f"Warmup batches: {args.warmup_batches}")
    print(f"{'='*60}")
    
    # Run benchmarks
    results = {}
    
    # 1. Baseline
    print(f"\n[1/4] Testing Baseline (no optimizations)...")
    model_baseline = AudioUNet1D(in_channels=1, base_channels=32, depth=4).to(device)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=3e-4)
    results['baseline'] = benchmark_baseline(
        model_baseline, dataloader, optimizer_baseline, device,
        args.num_batches, args.warmup_batches
    )
    
    # 2. Mixed Precision
    print(f"\n[2/4] Testing Mixed Precision (AMP)...")
    model_amp = AudioUNet1D(in_channels=1, base_channels=32, depth=4).to(device)
    optimizer_amp = torch.optim.AdamW(model_amp.parameters(), lr=3e-4)
    results['amp'] = benchmark_amp(
        model_amp, dataloader, optimizer_amp, device,
        args.num_batches, args.warmup_batches
    )
    
    # 3. torch.compile()
    print(f"\n[3/4] Testing torch.compile()...")
    model_compile = AudioUNet1D(in_channels=1, base_channels=32, depth=4).to(device)
    optimizer_compile = torch.optim.AdamW(model_compile.parameters(), lr=3e-4)
    results['compile'] = benchmark_compile(
        model_compile, dataloader, optimizer_compile, device,
        args.num_batches, args.warmup_batches
    )
    
    # 4. Combined (AMP + compile)
    print(f"\n[4/4] Testing Combined (AMP + compile)...")
    model_combined = AudioUNet1D(in_channels=1, base_channels=32, depth=4).to(device)
    optimizer_combined = torch.optim.AdamW(model_combined.parameters(), lr=3e-4)
    results['combined'] = benchmark_combined(
        model_combined, dataloader, optimizer_combined, device,
        args.num_batches, args.warmup_batches
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    
    print_results("Baseline", results['baseline'])
    print_results("Mixed Precision (AMP)", results['amp'], results['baseline'])
    print_results("torch.compile()", results['compile'], results['baseline'])
    print_results("Combined (AMP + compile)", results['combined'], results['baseline'])
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Optimization':<30} {'Time (ms)':<12} {'Speedup':<10} {'Improvement'}")
    print(f"{'-'*60}")
    
    baseline_time = results['baseline']['avg_batch_time'] * 1000
    print(f"{'Baseline':<30} {baseline_time:<12.1f} {'1.00x':<10} {'0%'}")
    
    for name in ['amp', 'compile', 'combined']:
        time_ms = results[name]['avg_batch_time'] * 1000
        speedup = results['baseline']['avg_batch_time'] / results[name]['avg_batch_time']
        improvement = (1 - results[name]['avg_batch_time'] / results['baseline']['avg_batch_time']) * 100
        
        label = {
            'amp': 'Mixed Precision (AMP)',
            'compile': 'torch.compile()',
            'combined': 'AMP + compile'
        }[name]
        
        print(f"{label:<30} {time_ms:<12.1f} {speedup:<10.2f}x {improvement:.1f}%")
    
    print(f"{'='*60}\n")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print()


if __name__ == "__main__":
    main()
