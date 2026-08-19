# Action Recognition in Videos

Deep learning-based human action classification on UCF101 using 3D CNNs (R(2+1)D-18) with PyTorch.

## Overview

This project classifies human actions in short video clips using a pretrained R(2+1)D-18 model fine-tuned on the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset. The system processes raw video files through a preprocessing pipeline that extracts and augments temporal clips, then feeds them through a 3D convolutional network that captures both spatial and motion features.

**Key Results:**
- Top-1 Accuracy: **89.80%**
- Top-5 Accuracy: **98.84%**
- Macro F1-Score: **0.8932**
- Inference Time: **11.5 ms/clip** (Tesla T4)

## Solution Approach

### Architecture: R(2+1)D-18

Rather than standard 3D convolutions, R(2+1)D factorizes each 3D convolution into a 2D spatial convolution followed by a 1D temporal convolution. This:
- Doubles the number of nonlinearities, improving representational capacity
- Makes optimization easier compared to full 3D convolutions
- Achieves strong accuracy with fewer parameters

The model is initialized with **Kinetics-400 pretrained weights** and fine-tuned on UCF101 with a replaced classification head (512 → 101 classes with dropout).

### Video Preprocessing Pipeline

```
Video → Frame Extraction → Temporal Sampling (16 frames, stride 2)
      → Resize (128×171) → Crop (112×112) → Normalize (ImageNet stats)
```

- **Temporal sampling:** 16 frames with stride 2 = effective 32-frame window
- **Augmentation:** Random crop, horizontal flip, color jitter, temporal jitter
- **Normalization:** ImageNet mean/std (required for pretrained backbone)

### Training Strategy

- **Optimizer:** SGD with momentum (0.9) and weight decay (1e-4)
- **Scheduler:** Cosine annealing with 3-epoch linear warmup
- **Mixed precision (AMP):** Reduces VRAM usage for RTX 3050 (4GB)
- **Gradient accumulation:** 4 steps → effective batch size of 32

## Project Structure

```
action-recognition/
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   ├── dataset.py            # UCF101 dataset class & preprocessing
│   └── __init__.py
├── models/
│   ├── video_classifier.py   # R3D-18 / R(2+1)D-18 / MC3-18 models
│   └── __init__.py
├── scripts/
│   ├── download_ucf101.py    # Dataset download (cross-platform)
│   ├── download_ucf101.sh    # Dataset download (bash)
│   ├── train.py              # Training with AMP & checkpointing
│   ├── evaluate.py           # Full evaluation & metrics
│   └── inference.py          # Visual prediction demos
├── utils/
│   ├── setup.py              # Seed & device utilities
│   └── __init__.py
├── results/                  # Checkpoints, logs, evaluation outputs
├── notebooks/
│   └── Action_Recognition_Training.ipynb  # Full Colab notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Usage

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3050, 4GB VRAM)
- ~7 GB disk space for UCF101 dataset

### Installation

```bash
git clone https://github.com/JaiAdith/action-recognition.git
cd action-recognition

pip install -r requirements.txt
```

### Download Dataset

```bash
# Cross-platform (Windows/Linux/Mac)
python scripts/download_ucf101.py

# Linux/Mac only
bash scripts/download_ucf101.sh
```

This downloads UCF101 (~6.5 GB) and the official train/test split annotations. Requires `unrar`, `7-Zip`, or `patool` for RAR extraction.

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

Training logs are saved to `results/logs/training_log.csv`. Checkpoints are saved to `results/checkpoints/`.

To resume from a checkpoint:
```bash
python scripts/train.py --config configs/default.yaml --resume results/checkpoints/latest.pth
```

### Evaluate

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint results/checkpoints/best.pth
```

Outputs saved to `results/evaluation/`:
- `evaluation_summary.json` — Top-1/5 accuracy, macro F1, inference time
- `per_class_metrics.csv` — Precision, recall, F1 per class
- `confusion_matrix.png` — Full 101×101 confusion matrix
- `top_confused_pairs.png` — Most commonly confused class pairs

### Visualize Predictions

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint results/checkpoints/best.pth \
    --num_samples 10
```

Generates annotated prediction images in `results/visualizations/`.

## Experiments

| Model | Pretrained | Clip Length | Top-1 | Top-5 |
|-------|-----------|-------------|-------|-------|
| R(2+1)D-18 | Kinetics-400 | 16 | 89.80% | 98.84% |

## Dependencies

- PyTorch >= 2.0
- torchvision >= 0.15
- OpenCV >= 4.8
- scikit-learn >= 1.3
- matplotlib, seaborn, numpy, pyyaml

## Dataset

[UCF101](https://www.crcv.ucf.edu/data/UCF101.php) — 13,320 video clips across 101 human action categories. Videos are sourced from YouTube and cover actions like sports, playing instruments, and daily activities.

## Acknowledgements

- UCF101 dataset: Soomro, Zamir, Shah. "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild." CRCV-TR-12-01, 2012.
- R(2+1)D architecture: Tran et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR 2018.
- Pretrained weights from torchvision (Kinetics-400).

## License

This project is for educational and assessment purposes.
