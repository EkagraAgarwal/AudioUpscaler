#!/usr/bin/env python3
"""
Training script for audio super-resolution model.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import AudioUNet1D, AudioUNet1DSimple, count_parameters
from dataset import AudioUpscaleDataset, create_dataloaders
from utils import (
    set_seed, AverageMeter, EarlyStopping,
    multi_resolution_stft_loss, compute_snr,
    get_lr, save_audio
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio super-resolution model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (optimal: 16 for RX 7700S with new architecture)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--audio-dir", type=str, default="data/raw/fma_small", help="Audio directory")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers (optimal: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--audio-length", type=int, default=32768, help="Audio length in samples (optimal: 32768)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--bitrate", type=int, default=64, help="Target bitrate for compression (default: 64 for curriculum learning)")
    parser.add_argument("--lite", action="store_true", help="Use lite model")
    parser.add_argument("--use-memmap", action="store_true", help="Use memory-mapped WAV files (recommended)")
    parser.add_argument("--wav-dir", type=str, default="data/wav_cache", help="WAV cache directory")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for ~27%% speedup (ROCm 7+ only)")
    parser.add_argument("--dynamic-bitrate", action="store_true", help="Use dynamic bitrate (32, 48, 64, 96, 128 kbps) favoring lower values")
    parser.add_argument("--l1-weight", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument("--stft-weight", type=float, default=1.0, help="STFT loss weight")
    return parser.parse_args()


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    args
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    l1_meter = AverageMeter()
    stft_meter = AverageMeter()
    snr_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    l1_weight = args.l1_weight if hasattr(args, 'l1_weight') else 1.0
    stft_weight = args.stft_weight if hasattr(args, 'stft_weight') else 1.0

    for batch_idx, (compressed, original) in enumerate(pbar):
        compressed = compressed.to(device)
        original = original.to(device)

        optimizer.zero_grad()

        output = model(compressed)

        l1_loss = nn.functional.l1_loss(output, original)
        stft_loss = multi_resolution_stft_loss(output, original)
        loss = l1_weight * l1_loss + stft_weight * stft_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            snr = compute_snr(output, original)

        loss_meter.update(loss.item(), compressed.size(0))
        l1_meter.update(l1_loss.item(), compressed.size(0))
        stft_meter.update(stft_loss.item(), compressed.size(0))
        snr_meter.update(snr, compressed.size(0))

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'l1': f'{l1_meter.avg:.4f}',
            'stft': f'{stft_meter.avg:.4f}',
            'snr': f'{snr_meter.avg:.2f}'
        })

        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/l1_loss', l1_loss.item(), global_step)
        writer.add_scalar('train/stft_loss', stft_loss.item(), global_step)
        writer.add_scalar('train/snr', snr, global_step)
        writer.add_scalar('train/lr', get_lr(optimizer), global_step)

    return loss_meter.avg, snr_meter.avg


def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    writer,
    args
):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    snr_meter = AverageMeter()
    
    with torch.no_grad():
        for compressed, original in tqdm(val_loader, desc="Validating"):
            compressed = compressed.to(device)
            original = original.to(device)
            
            output = model(compressed)
            
            l1_loss = nn.functional.l1_loss(output, original)
            stft_loss = multi_resolution_stft_loss(output, original)
            loss = l1_loss + stft_loss
            
            snr = compute_snr(output, original)
            
            loss_meter.update(loss.item(), compressed.size(0))
            snr_meter.update(snr, compressed.size(0))
    
    writer.add_scalar('val/loss', loss_meter.avg, epoch)
    writer.add_scalar('val/snr', snr_meter.avg, epoch)
    
    print(f"Validation - Loss: {loss_meter.avg:.4f}, SNR: {snr_meter.avg:.2f} dB")
    
    return loss_meter.avg, snr_meter.avg


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print("\nCreating dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        audio_length=args.audio_length,
        sample_rate=args.sample_rate,
        bitrate=args.bitrate,
        num_workers=args.num_workers,
        use_memmap=args.use_memmap,
        wav_dir=args.wav_dir if args.use_memmap else None,
        dynamic_bitrate=args.dynamic_bitrate
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\nCreating model...")
    if args.lite:
        model = AudioUNet1DSimple(in_channels=1, channels=32, use_interpolation_upsampling=True)
    else:
        model = AudioUNet1D(in_channels=1, base_channels=32, depth=4, use_dilated_bottleneck=True, use_interpolation_upsampling=True)

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("Model compiled successfully!")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
    
    print(f"\nTraining for {args.epochs} epoch(s)...")

    l1_weight = args.l1_weight if hasattr(args, 'l1_weight') else 1.0
    stft_weight = args.stft_weight if hasattr(args, 'stft_weight') else 1.0

    for epoch in range(start_epoch, args.epochs):
        if epoch < 15:
            args.l1_weight = 10.0
            args.stft_weight = 0.1
        elif epoch < 30:
            args.l1_weight = 5.0
            args.stft_weight = 0.5
        else:
            args.l1_weight = l1_weight
            args.stft_weight = stft_weight

        train_loss, train_snr = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, writer, args
        )

        val_loss, val_snr = validate(
            model, val_loader, criterion, device, epoch, writer, args
        )

        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                          checkpoint_dir / "best_model.pt")
            print(f"Saved best model (loss: {val_loss:.4f})")
        
        save_checkpoint(model, optimizer, epoch, val_loss,
                       checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss,
                   checkpoint_dir / "final_model.pt")
    
    writer.close()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
