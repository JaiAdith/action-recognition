"""
Video Classification Models for Action Recognition.

Supports three 3D CNN architectures from torchvision:
- R3D-18: Standard 3D ResNet-18
- R(2+1)D-18: Factored 3D convolutions (spatial + temporal)
- MC3-18: Mixed 3D/2D convolution ResNet

All models use Kinetics-400 pretrained weights and replace the final FC
layer for UCF101 (101 classes).
"""

import torch
import torch.nn as nn
from torchvision.models.video import (
    r3d_18,
    r2plus1d_18,
    mc3_18,
    R3D_18_Weights,
    R2Plus1D_18_Weights,
    MC3_18_Weights,
)


MODEL_REGISTRY = {
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
}


class ActionRecognitionModel(nn.Module):
    """
    Wrapper around torchvision 3D CNN models for action recognition.
    
    Loads a pretrained backbone, replaces the classification head,
    and optionally adds dropout for regularization.
    
    Args:
        architecture: One of 'r3d_18', 'r2plus1d_18', 'mc3_18'.
        num_classes: Number of output action classes.
        pretrained: Whether to load Kinetics-400 pretrained weights.
        dropout: Dropout probability before the final FC layer.
        freeze_bn: Whether to freeze batch normalization layers.
    """

    def __init__(
        self,
        architecture: str = "r2plus1d_18",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_bn: bool = False,
    ):
        super().__init__()

        if architecture not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Choose from {list(MODEL_REGISTRY.keys())}"
            )

        model_fn, weights_cls = MODEL_REGISTRY[architecture]

        # Load model with or without pretrained weights
        if pretrained:
            print(f"Loading {architecture} with Kinetics-400 pretrained weights")
            self.backbone = model_fn(weights=weights_cls)
        else:
            print(f"Loading {architecture} from scratch (no pretraining)")
            self.backbone = model_fn(weights=None)

        # Get feature dimension from original FC layer
        in_features = self.backbone.fc.in_features

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.freeze_bn = freeze_bn
        self.architecture = architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W).
               B = batch, C = 3 (RGB), T = clip_length, H/W = crop_size.
               
        Returns:
            logits: (B, num_classes) raw classification scores.
        """
        return self.backbone(x)

    def train(self, mode: bool = True):
        """Override to optionally freeze BN layers during training."""
        super().train(mode)
        if self.freeze_bn and mode:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        return self


def build_model(cfg: dict) -> ActionRecognitionModel:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = ActionRecognitionModel(
        architecture=model_cfg["architecture"],
        num_classes=data_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
        freeze_bn=model_cfg["freeze_bn"],
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model
