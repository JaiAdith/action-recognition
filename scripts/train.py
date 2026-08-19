"""
Training script for Action Recognition model.

Features:
- Mixed precision training (AMP) for memory efficiency on RTX 3050
- Gradient accumulation for larger effective batch size
- Cosine annealing LR scheduler with warmup
- Checkpointing (saves best + latest)
- TensorBoard-compatible CSV logging

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import build_dataloaders
from models.video_classifier import build_model
from utils.setup import set_seed, get_device


def get_scheduler(optimizer, cfg, steps_per_epoch):
    """Build LR scheduler with optional warmup.

    steps_per_epoch should be the number of *optimizer* steps per epoch
    (i.e. batches // gradient_accumulation_steps), matching when
    scheduler.step() is actually called.
    """
    train_cfg = cfg["training"]
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = train_cfg["warmup_epochs"] * steps_per_epoch

    if train_cfg["scheduler"] == "cosine":
        # Cosine annealing after warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler,
    device, epoch, cfg, logger
):
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    use_amp = cfg["training"]["mixed_precision"]

    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    for batch_idx, (clips, labels) in enumerate(pbar):
        start_time = time.time()

        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with AMP
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels) / accum_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Metrics
        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        avg_loss = running_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%", lr=f"{lr:.6f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    avg_batch_time = sum(batch_times) / len(batch_times)

    return epoch_loss, epoch_acc, avg_batch_time


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=True):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for clips, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        running_loss += loss.item()

        # Top-1 accuracy
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct_top1 += pred.eq(labels).sum().item()

        # Top-5 accuracy
        _, pred5 = outputs.topk(5, dim=1)
        correct_top5 += sum(
            labels[i] in pred5[i] for i in range(labels.size(0))
        )

    avg_loss = running_loss / len(dataloader)
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    return avg_loss, top1_acc, top5_acc


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Action Recognition Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup
    set_seed(cfg["seed"])
    device = get_device()

    # Create output directories
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    log_dir = Path(cfg["training"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build data, model, optimizer
    print("\n=== Building Dataset ===")
    train_loader, test_loader, num_classes = build_dataloaders(cfg)

    print("\n=== Building Model ===")
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    accum = cfg["training"]["gradient_accumulation_steps"]
    optim_steps_per_epoch = (len(train_loader) + accum - 1) // accum
    scheduler = get_scheduler(optimizer, cfg, optim_steps_per_epoch)
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["metrics"].get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")

    # CSV logger
    log_file = log_dir / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "test_loss",
            "test_top1", "test_top5", "lr", "batch_time"
        ])

    # Training loop
    print("\n=== Training ===")
    print(f"Epochs: {cfg['training']['epochs']}")
    print(f"Batch size: {cfg['training']['batch_size']} "
          f"(effective: {cfg['training']['batch_size'] * cfg['training']['gradient_accumulation_steps']})")
    print(f"Mixed precision: {cfg['training']['mixed_precision']}")
    print()

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, batch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, cfg, None
        )

        # Evaluate
        test_loss, top1_acc, top5_acc = evaluate(
            model, test_loader, criterion, device,
            use_amp=cfg["training"]["mixed_precision"]
        )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch}/{cfg['training']['epochs']-1} ({epoch_time:.0f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Top-1: {top1_acc:.2f}% Top-5: {top5_acc:.2f}%\n"
        )

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{test_loss:.4f}", f"{top1_acc:.2f}", f"{top5_acc:.2f}",
                f"{lr:.6f}", f"{batch_time:.3f}"
            ])

        # Save checkpoints
        metrics = {
            "train_loss": train_loss, "train_acc": train_acc,
            "test_top1": top1_acc, "test_top5": top5_acc,
            "best_acc": max(best_acc, top1_acc),
        }

        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, metrics,
            ckpt_dir / "latest.pth"
        )

        if top1_acc > best_acc:
            best_acc = top1_acc
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, metrics,
                ckpt_dir / "best.pth"
            )
            print(f"  >> New best model saved: {top1_acc:.2f}%")

    print(f"\n=== Training Complete ===")
    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
