import math

import torch
from torch import nn


class ECA(nn.Module):
    """Efficient Channel Attention: learns how much weight each channel deserves."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 == 1 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)                       # (B, C, 1, 1)  one number per channel
        w = w.squeeze(-1).transpose(1, 2)      # (B, 1, C)
        w = torch.sigmoid(self.conv(w))        # (B, 1, C)     weights between 0 and 1
        w = w.transpose(1, 2).unsqueeze(-1)    # (B, C, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, optional ECA, then 2x2 max pooling."""

    def __init__(self, c_in: int, c_out: int, use_eca: bool):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c_out) if use_eca else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.eca(self.body(x)))


class GenreCNN(nn.Module):
    def __init__(self, n_classes: int = 10, use_eca: bool = True,
                 widths: tuple = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, c = [], 1
        for w in widths:
            blocks.append(ConvBlock(c, w, use_eca))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector before the classifier (256 numbers); may be useful later."""
        return self.features(x).mean(dim=(2, 3))