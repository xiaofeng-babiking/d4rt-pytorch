#!/usr/bin/env python3
"""
Training script for D4RT model.

Supports:
- Gradient accumulation for simulating larger batch sizes
- Mixed precision training (AMP)
- Gradient checkpointing for memory optimization
- torch.compile for PyTorch 2.0+
- Distributed training (DDP)
- TensorBoard logging
- Automatic checkpoint resumption

Usage:
    # Single GPU with gradient accumulation
    python train.py --config configs/d4rt_rtx5090.yaml --data-root /path/to/data

    # Multi-GPU
    torchrun --nproc_per_node=8 train.py --config configs/d4rt_large.yaml --data-root /path/to/data
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml
import json
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from models import D4RT, create_d4rt
from losses import D4RTLoss
from data import (
    PointOdysseyDataset,
    VideoDataset,
    KubricDataset,
    SintelDataset,
    ScanNetDataset,
    collate_fn,
)
from data.augmentations import VideoAugmentation, TemporalSubsampling, AugmentationConfig

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train D4RT model')

    # Model
    parser.add_argument('--encoder', type=str, default='base',
                        choices=['base', 'large', 'huge', 'giant'],
                        help='Encoder variant')
    parser.add_argument('--decoder-depth', type=int, default=8,
                        help='Number of decoder layers')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size. Must be 224 when using the '
                             'pretrained VideoMAE-base encoder (fixed 14x14 '
                             'patch grid). Set to 256 only with a from-scratch '
                             'encoder.')
    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames per clip')
    parser.add_argument('--patch-size', type=int, default=9,
                        help='Local RGB patch size for queries')

    # Training
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps (effective_batch = batch_size * accum_steps * num_gpus)')
    parser.add_argument('--num-queries', type=int, default=2048,
                        help='Number of queries per batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--steps', type=int, default=500000,
                        help='Total training steps (optimizer steps, not micro-batches)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2500,
                        help='Warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.03,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping (L2 norm)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    # Memory optimization
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+) for faster training')

    # Loss weights
    parser.add_argument('--lambda-3d', type=float, default=1.0)
    parser.add_argument('--lambda-2d', type=float, default=0.1)
    parser.add_argument('--lambda-vis', type=float, default=0.1)
    parser.add_argument('--lambda-disp', type=float, default=0.1)
    parser.add_argument('--lambda-normal', type=float, default=0.5)
    parser.add_argument('--lambda-conf', type=float, default=0.2)

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to data root')
    parser.add_argument('--dataset', type=str, default='video',
                        choices=['video', 'kubric', 'sintel', 'scannet', 'pointodyssey'],
                        help='Dataset type')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Save checkpoint frequency (steps)')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Logging frequency (steps)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--auto-resume', action='store_true',
                        help='Automatically resume from latest checkpoint if exists')
    parser.add_argument('--pretrained-encoder', type=str, default=None,
                        help='Path to pretrained encoder weights')

    # Distributed
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            # Convert hyphens to underscores for attribute names
            attr_name = key.replace('-', '_')
            if hasattr(args, attr_name):
                # Only override if not explicitly set via command line
                default_val = parser.get_default(attr_name)
                current_val = getattr(args, attr_name)
                if current_val == default_val:
                    setattr(args, attr_name, value)

    return args


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        rank = 0
        world_size = 1
        local_rank = 0
    else:
        return 0, 1, 0

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    return rank, world_size, local_rank


def create_dataloader(args, rank, world_size):
    """Create training dataloader."""
    # Augmentation
    aug_config = AugmentationConfig()
    transform = VideoAugmentation(aug_config)

    # Dataset
    if args.dataset == 'pointodyssey':
        dataset = PointOdysseyDataset(
            args.data_root,
            split='train',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )
    elif args.dataset == 'kubric':
        dataset = KubricDataset(
            args.data_root,
            split='train',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )
    elif args.dataset == 'sintel':
        dataset = SintelDataset(
            args.data_root,
            split='training',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )
    elif args.dataset == 'scannet':
        dataset = ScanNetDataset(
            args.data_root,
            split='train',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )
    else:
        dataset = VideoDataset(
            args.data_root,
            split='train',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )

    # Sampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0
    )

    return dataloader, sampler


def create_optimizer_scheduler(model, args, total_steps):
    """Create optimizer and learning rate scheduler."""
    # Separate weight decay for different parameter groups
    # Don't apply weight decay to bias and LayerNorm
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name.lower() or 'ln' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Cosine scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            # Linear warmup
            return step / max(1, args.warmup_steps)
        else:
            # Cosine decay
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def forward_backward_step(
    model,
    batch,
    criterion,
    scaler,
    args,
    device,
    is_accumulating: bool
):
    """
    Single forward-backward step (micro-batch).

    Args:
        model: The model
        batch: Input batch
        criterion: Loss function
        scaler: GradScaler for AMP
        args: Training arguments
        device: Device to use
        is_accumulating: Whether this is an accumulation step (don't sync gradients in DDP)

    Returns:
        Dictionary of loss values
    """
    # Move to device
    video = batch['video'].to(device, non_blocking=True)
    coords = batch['coords'].to(device, non_blocking=True)
    t_src = batch['t_src'].to(device, non_blocking=True)
    t_tgt = batch['t_tgt'].to(device, non_blocking=True)
    t_cam = batch['t_cam'].to(device, non_blocking=True)
    aspect_ratio = batch['aspect_ratio'].to(device, non_blocking=True)

    targets = {k: v.to(device, non_blocking=True) for k, v in batch['targets'].items()}

    # Context manager for gradient accumulation (disable gradient sync during accumulation)
    if is_accumulating and hasattr(model, 'no_sync'):
        context = model.no_sync()
    else:
        context = nullcontext()

    with context:
        # Forward pass with autocast
        with autocast(enabled=args.amp):
            # Reshape video for model: (B, T, H, W, C) -> (B, C, T, H, W)
            video_input = video.permute(0, 4, 1, 2, 3)

            predictions = model(video_input, coords, t_src, t_tgt, t_cam, aspect_ratio)
            losses = criterion(predictions, targets)

            # Scale loss by accumulation steps
            loss = losses['loss'] / args.gradient_accumulation_steps

        # Backward pass
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Return unscaled losses for logging
    return {k: v.item() for k, v in losses.items()}


def optimizer_step(model, optimizer, scheduler, scaler, args):
    """Perform optimizer step with gradient clipping."""
    if args.amp and scaler is not None:
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
    else:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

    # LR scheduler step
    scheduler.step()

    # Zero gradients for next iteration
    optimizer.zero_grad(set_to_none=True)


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, args, output_dir, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'args': vars(args)
    }

    checkpoint_path = output_dir / f'checkpoint_{step:08d}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest
    latest_path = output_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = output_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('step', 0), checkpoint.get('epoch', 0)


def nullcontext():
    """Context manager that does nothing (for Python < 3.7 compatibility)."""
    class NullContext:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    return NullContext()


def get_grad_norm(model):
    """Compute gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def format_time(seconds):
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    # Setup device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size

    if is_main:
        print("=" * 60)
        print("D4RT Training")
        print("=" * 60)
        print(f"Encoder: {args.encoder}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Mixed precision: {args.amp}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"torch.compile: {args.compile}")
        print("=" * 60)

    # Create model
    model = create_d4rt(
        variant=args.encoder,
        img_size=args.img_size,
        temporal_size=args.num_frames,
        decoder_depth=args.decoder_depth,
        query_patch_size=args.patch_size
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()
            if is_main:
                print("Gradient checkpointing enabled")

    # Load pretrained encoder if provided
    if args.pretrained_encoder:
        if is_main:
            print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        checkpoint = torch.load(args.pretrained_encoder, map_location='cpu')
        model.encoder.load_state_dict(checkpoint, strict=False)

    model = model.to(device)

    # Compile model for faster training (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # DDP wrapper
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Create criterion
    criterion = D4RTLoss(
        lambda_3d=args.lambda_3d,
        lambda_2d=args.lambda_2d,
        lambda_vis=args.lambda_vis,
        lambda_disp=args.lambda_disp,
        lambda_normal=args.lambda_normal,
        lambda_conf=args.lambda_conf
    )

    # Create dataloader
    dataloader, sampler = create_dataloader(args, rank, world_size)
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = args.steps

    if is_main:
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, args, total_steps)

    # AMP scaler
    scaler = GradScaler() if args.amp else None

    # TensorBoard writer
    writer = None
    if is_main and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_dir / 'tensorboard')

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0

    # Auto-resume from latest checkpoint
    if args.auto_resume and (output_dir / 'checkpoint_latest.pth').exists():
        args.resume = str(output_dir / 'checkpoint_latest.pth')

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        start_step, start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # Training loop
    if is_main:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Starting training from step {start_step}")

    model.train()
    step = start_step
    epoch = start_epoch
    running_loss = {}
    accum_count = 0
    batch_start_time = time.time()
    train_start_time = time.time()

    # Zero gradients at start
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        if sampler:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if step >= total_steps:
                break

            # Determine if this is an accumulation step
            accum_count += 1
            is_accumulating = (accum_count % args.gradient_accumulation_steps != 0)

            # Forward-backward
            losses = forward_backward_step(
                model, batch, criterion, scaler, args, device, is_accumulating
            )

            # Accumulate losses for logging
            for k, v in losses.items():
                if k not in running_loss:
                    running_loss[k] = 0
                running_loss[k] += v / args.gradient_accumulation_steps

            # Optimizer step after accumulation
            if not is_accumulating:
                optimizer_step(model, optimizer, scheduler, scaler, args)
                step += 1

                # Logging
                if is_main and step % args.log_freq == 0:
                    batch_time = time.time() - batch_start_time
                    samples_per_sec = args.log_freq * effective_batch_size / batch_time
                    elapsed_time = time.time() - train_start_time
                    eta_seconds = (total_steps - step) / step * elapsed_time if step > 0 else 0

                    lr = scheduler.get_last_lr()[0]
                    loss_str = ' | '.join([f'{k}: {v / args.log_freq:.4f}' for k, v in running_loss.items()])

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[{ts}] "
                        f"Step {step}/{total_steps} | "
                        f"Epoch {epoch} | "
                        f"LR: {lr:.2e} | "
                        f"{loss_str} | "
                        f"Samples/s: {samples_per_sec:.1f} | "
                        f"ETA: {format_time(eta_seconds)}"
                    )

                    # TensorBoard logging
                    if writer:
                        writer.add_scalar('train/lr', lr, step)
                        writer.add_scalar('train/samples_per_sec', samples_per_sec, step)
                        for k, v in running_loss.items():
                            writer.add_scalar(f'train/{k}', v / args.log_freq, step)

                    running_loss = {}
                    batch_start_time = time.time()

                # Save checkpoint
                if is_main and step % args.save_freq == 0:
                    checkpoint_path = save_checkpoint(
                        model, optimizer, scheduler, scaler, step, epoch, args, output_dir
                    )
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts}] Saved checkpoint to {checkpoint_path}")

                    if writer:
                        writer.flush()

        epoch += 1

    # Final checkpoint
    if is_main:
        checkpoint_path = save_checkpoint(
            model, optimizer, scheduler, scaler, step, epoch, args, output_dir
        )
        print(f"\nTraining complete!")
        print(f"Final checkpoint: {checkpoint_path}")
        print(f"Total time: {format_time(time.time() - train_start_time)}")

        if writer:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
