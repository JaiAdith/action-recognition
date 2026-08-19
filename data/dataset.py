"""
UCF101 Dataset class with video preprocessing pipeline.

Handles:
- Parsing official train/test split files
- Frame extraction from video files using OpenCV
- Temporal sampling with configurable stride and jitter
- Spatial augmentations (resize, crop, flip, color jitter)
- Normalization to ImageNet statistics (for pretrained models)
"""

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ImageNet normalization (used by Kinetics-400 pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UCF101Dataset(Dataset):
    """
    UCF101 Action Recognition Dataset.
    
    Loads video clips and applies spatial/temporal transforms for training
    or evaluation of 3D CNN models.
    
    Args:
        root_dir: Path to UCF-101 video directory (contains class folders).
        annotation_dir: Path to ucfTrainTestlist directory.
        split: Which train/test split to use (1, 2, or 3).
        train: If True, load training set; otherwise test set.
        clip_length: Number of frames to sample per clip.
        frame_stride: Sample every Nth frame from video.
        resize: (H, W) to resize frames before cropping.
        crop_size: Spatial crop size.
        augment: Whether to apply data augmentation.
        temporal_jitter: Randomly offset the clip start position.
        color_jitter: Dict with brightness, contrast, saturation, hue params.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_dir: str,
        split: int = 1,
        train: bool = True,
        clip_length: int = 16,
        frame_stride: int = 2,
        resize: tuple = (128, 171),
        crop_size: int = 112,
        augment: bool = True,
        temporal_jitter: bool = True,
        color_jitter: Optional[dict] = None,
    ):
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.resize = resize
        self.crop_size = crop_size
        self.augment = augment and train
        self.temporal_jitter = temporal_jitter and train
        self.color_jitter = color_jitter if train else None
        self.train = train

        # Parse class list and annotations
        self.classes, self.class_to_idx = self._parse_classes(annotation_dir)
        self.samples = self._parse_split(annotation_dir, split, train)
        
        print(
            f"{'Train' if train else 'Test'} split {split}: "
            f"{len(self.samples)} videos, {len(self.classes)} classes"
        )

    def _parse_classes(self, annotation_dir: str):
        """Parse classInd.txt to get class names and indices."""
        class_file = Path(annotation_dir) / "classInd.txt"
        classes = {}
        with open(class_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                classes[name] = int(idx) - 1  # 0-indexed
        class_names = sorted(classes.keys(), key=lambda x: classes[x])
        return class_names, classes

    def _parse_split(self, annotation_dir: str, split: int, train: bool):
        """Parse train/test split files to get video paths and labels."""
        ann_dir = Path(annotation_dir)
        if train:
            filename = f"trainlist{split:02d}.txt"
        else:
            filename = f"testlist{split:02d}.txt"

        samples = []
        with open(ann_dir / filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                video_path = parts[0]  # e.g., "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
                class_name = video_path.split("/")[0]
                label = self.class_to_idx[class_name]
                full_path = self.root_dir / video_path
                if full_path.exists():
                    samples.append((str(full_path), label))

        return samples

    def _load_frames(self, video_path: str) -> np.ndarray:
        """
        Load and sample frames from a video file.

        Uses sequential reading with grab/retrieve instead of repeated seeks,
        which avoids costly keyframe-to-target decoding on every frame.

        Returns:
            frames: np.ndarray of shape (T, H, W, C) in uint8 RGB.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        required_span = self.clip_length * self.frame_stride

        if total_frames <= required_span:
            start = 0
        elif self.temporal_jitter:
            start = random.randint(0, total_frames - required_span)
        else:
            start = (total_frames - required_span) // 2

        frame_indices = set(
            min(start + i * self.frame_stride, total_frames - 1)
            for i in range(self.clip_length)
        )
        last_needed = max(frame_indices)

        # Seek once to start, then read sequentially
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        grabbed = {}
        for idx in range(start, last_needed + 1):
            ret = cap.grab()
            if not ret:
                break
            if idx in frame_indices:
                ret2, frame = cap.retrieve()
                if ret2:
                    grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()

        # Assemble frames in order
        frames = []
        ordered_indices = [
            min(start + i * self.frame_stride, total_frames - 1)
            for i in range(self.clip_length)
        ]
        for idx in ordered_indices:
            if idx in grabbed:
                frames.append(grabbed[idx])
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))

        while len(frames) < self.clip_length:
            frames.append(frames[-1].copy())

        return np.stack(frames)

    def _apply_transforms(self, frames: np.ndarray) -> torch.Tensor:
        """
        Apply spatial transforms to a clip.
        
        Args:
            frames: (T, H, W, C) uint8 RGB array.
            
        Returns:
            tensor: (C, T, H, W) float32 normalized tensor.
        """
        T, H, W, C = frames.shape

        # Resize all frames
        resized = np.stack([
            cv2.resize(f, (self.resize[1], self.resize[0]))
            for f in frames
        ])  # (T, resize_H, resize_W, C)

        rH, rW = self.resize

        # Spatial crop
        if self.augment:
            # Random crop
            y = random.randint(0, rH - self.crop_size)
            x = random.randint(0, rW - self.crop_size)
        else:
            # Center crop
            y = (rH - self.crop_size) // 2
            x = (rW - self.crop_size) // 2

        cropped = resized[:, y:y + self.crop_size, x:x + self.crop_size, :]

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            cropped = cropped[:, :, ::-1, :].copy()

        # Color jitter (applied per-clip, same transform for all frames)
        if self.color_jitter:
            cropped = self._color_jitter(cropped)

        # Convert to float and normalize
        tensor = torch.from_numpy(cropped).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 1, 3)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 1, 3)
        tensor = (tensor - mean) / std

        # Rearrange: (T, H, W, C) -> (C, T, H, W)
        tensor = tensor.permute(3, 0, 1, 2)

        return tensor

    def _color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Apply random color jittering consistently across all frames."""
        brightness = 1.0 + random.uniform(
            -self.color_jitter.get("brightness", 0),
            self.color_jitter.get("brightness", 0),
        )
        contrast = 1.0 + random.uniform(
            -self.color_jitter.get("contrast", 0),
            self.color_jitter.get("contrast", 0),
        )

        # Apply brightness and contrast
        frames = frames.astype(np.float32)
        frames = frames * brightness
        mean = frames.mean(axis=(1, 2), keepdims=True)
        frames = (frames - mean) * contrast + mean
        frames = np.clip(frames, 0, 255).astype(np.uint8)

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        clip = self._apply_transforms(frames)
        return clip, label


def build_dataloaders(cfg: dict):
    """
    Build train and test dataloaders from config.
    
    Args:
        cfg: Parsed YAML config dict.
        
    Returns:
        train_loader, test_loader, num_classes
    """
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})

    train_dataset = UCF101Dataset(
        root_dir=data_cfg["root_dir"],
        annotation_dir=data_cfg["annotation_dir"],
        split=data_cfg["split"],
        train=True,
        clip_length=data_cfg["clip_length"],
        frame_stride=data_cfg["frame_stride"],
        resize=tuple(data_cfg["resize"]),
        crop_size=data_cfg["crop_size"],
        augment=True,
        temporal_jitter=aug_cfg.get("temporal_jitter", True),
        color_jitter={
            k: v for k, v in aug_cfg.get("color_jitter", {}).items()
        } if "color_jitter" in aug_cfg else None,
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, test_loader, data_cfg["num_classes"]
