"""
Comprehensive evaluation script for Action Recognition model.

Generates:
- Top-1 and Top-5 accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (saved as image)
- Inference time per video clip
- Classification report saved to CSV

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint results/checkpoints/best.pth
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import build_dataloaders
from models.video_classifier import build_model
from utils.setup import set_seed, get_device


@torch.no_grad()
def run_evaluation(model, dataloader, device, use_amp=True):
    """Run inference on entire test set, collecting predictions and timing."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    for clips, labels in tqdm(dataloader, desc="Running evaluation"):
        clips = clips.to(device, non_blocking=True)

        # Time inference
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(clips)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        inference_times.append(elapsed / clips.size(0))  # per-clip time

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        inference_times,
    )


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    # For 101 classes, use a large figure and skip tick labels
    fig, ax = plt.subplots(figsize=(20, 18))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Confusion Matrix - UCF101 Action Recognition", fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_top_confused(cm, class_names, save_path, top_k=15):
    """Plot the most confused class pairs."""
    # Zero out diagonal
    cm_off = cm.copy().astype(float)
    np.fill_diagonal(cm_off, 0)

    # Find top confused pairs
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_off[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm_off[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_k]

    if not pairs:
        return

    labels = [f"{p[0]} → {p[1]}" for p in pairs]
    values = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(labels)), values, color="#4c72b0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Misclassifications")
    ax.set_title(f"Top {top_k} Most Confused Class Pairs")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Top confused pairs saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Action Recognition Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data and model
    print("=== Loading Data ===")
    _, test_loader, num_classes = build_dataloaders(cfg)

    print("\n=== Loading Model ===")
    model = build_model(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Run inference
    print("\n=== Running Evaluation ===")
    preds, labels, probs, times = run_evaluation(
        model, test_loader, device,
        use_amp=cfg["training"]["mixed_precision"],
    )

    # Get class names from dataset
    class_names = test_loader.dataset.classes

    # ---- Metrics ----
    top1_acc = 100.0 * np.mean(preds == labels)
    top5_acc = 100.0 * top_k_accuracy_score(labels, probs, k=5, labels=range(num_classes))
    avg_inference_time = np.mean(times) * 1000  # ms

    print(f"\n{'='*50}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Avg Inference Time: {avg_inference_time:.1f} ms/clip")
    print(f"{'='*50}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(num_classes), zero_division=0
    )

    # Save per-class report to CSV
    report_path = output_dir / "per_class_metrics.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for i, name in enumerate(class_names):
            writer.writerow([
                name,
                f"{precision[i]:.4f}",
                f"{recall[i]:.4f}",
                f"{f1[i]:.4f}",
                int(support[i]),
            ])
        # Macro averages
        writer.writerow([
            "MACRO_AVG",
            f"{precision.mean():.4f}",
            f"{recall.mean():.4f}",
            f"{f1.mean():.4f}",
            int(support.sum()),
        ])
    print(f"Per-class metrics saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    plot_top_confused(cm, class_names, output_dir / "top_confused_pairs.png")

    # Save summary
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": int(checkpoint["epoch"]),
        "top1_accuracy": round(top1_acc, 2),
        "top5_accuracy": round(top5_acc, 2),
        "avg_inference_time_ms": round(avg_inference_time, 1),
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "num_test_samples": int(len(labels)),
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary saved to {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
