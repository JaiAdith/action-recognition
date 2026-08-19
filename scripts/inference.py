"""
Inference script for visualizing model predictions on sample videos.

Generates:
- Annotated frames with predicted action labels and confidence
- Top-5 prediction bar charts per video
- Saves results as images for the report

Usage:
    python scripts/inference.py \
        --config configs/default.yaml \
        --checkpoint results/checkpoints/best.pth \
        --video_dir data/UCF-101 \
        --num_samples 10
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import UCF101Dataset
from models.video_classifier import build_model
from utils.setup import set_seed, get_device


def visualize_prediction(
    video_path: str,
    true_label: str,
    pred_label: str,
    top5_classes: list,
    top5_probs: list,
    frames: np.ndarray,
    save_path: str,
):
    """
    Create a visualization with sample frames and prediction bar chart.
    
    Layout: top row = 4 sampled frames, bottom = top-5 prediction bar chart.
    """
    fig = plt.figure(figsize=(14, 7))
    
    # Display 4 evenly spaced frames
    num_display = min(4, len(frames))
    indices = np.linspace(0, len(frames) - 1, num_display, dtype=int)
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, num_display, i + 1)
        ax.imshow(frames[idx])
        ax.set_title(f"Frame {idx}", fontsize=10)
        ax.axis("off")

    # Top-5 prediction bar chart
    ax_bar = fig.add_subplot(2, 1, 2)
    colors = ["#2ecc71" if c == true_label else "#e74c3c" for c in top5_classes]
    bars = ax_bar.barh(range(len(top5_classes)), top5_probs, color=colors)
    ax_bar.set_yticks(range(len(top5_classes)))
    ax_bar.set_yticklabels(top5_classes, fontsize=10)
    ax_bar.set_xlabel("Confidence", fontsize=11)
    ax_bar.set_xlim(0, 1)
    ax_bar.invert_yaxis()

    # Add confidence percentages on bars
    for bar, prob in zip(bars, top5_probs):
        ax_bar.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", fontsize=10
        )

    correct = "✓" if pred_label == true_label else "✗"
    video_name = Path(video_path).stem
    fig.suptitle(
        f"{video_name}\nTrue: {true_label} | Predicted: {pred_label} {correct}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_inference(model, dataset, device, num_samples, output_dir, use_amp=True):
    """Run inference on random samples and generate visualizations."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    class_names = dataset.classes

    results = []

    for i, idx in enumerate(indices):
        video_path, true_idx = dataset.samples[idx]
        true_label = class_names[true_idx]

        # Load raw frames for visualization (before transforms)
        raw_frames = dataset._load_frames(video_path)  # (T, H, W, C) uint8 RGB

        # Get model input
        clip, label = dataset[idx]
        clip = clip.unsqueeze(0).to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            output = model(clip)

        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_classes = [class_names[j] for j in top5_idx]
        top5_probs = [probs[j] for j in top5_idx]
        pred_label = top5_classes[0]

        save_path = output_dir / f"prediction_{i+1:02d}_{Path(video_path).stem}.png"
        visualize_prediction(
            video_path, true_label, pred_label,
            top5_classes, top5_probs, raw_frames, str(save_path)
        )

        status = "CORRECT" if pred_label == true_label else "WRONG"
        print(f"[{i+1}/{len(indices)}] {status}: {true_label} → {pred_label} ({top5_probs[0]:.1%})")

        results.append({
            "video": Path(video_path).name,
            "true": true_label,
            "predicted": pred_label,
            "confidence": top5_probs[0],
            "correct": pred_label == true_label,
        })

    # Summary
    correct_count = sum(r["correct"] for r in results)
    print(f"\nSample accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.0f}%)")
    print(f"Visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run inference and visualize predictions")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/visualizations")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()

    # Build model
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Build test dataset (no augmentation)
    data_cfg = cfg["data"]
    test_dataset = UCF101Dataset(
        root_dir=data_cfg["root_dir"],
        annotation_dir=data_cfg["annotation_dir"],
        split=data_cfg["split"],
        train=False,
        clip_length=data_cfg["clip_length"],
        frame_stride=data_cfg["frame_stride"],
        resize=tuple(data_cfg["resize"]),
        crop_size=data_cfg["crop_size"],
        augment=False,
        temporal_jitter=False,
    )

    run_inference(model, test_dataset, device, args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()
