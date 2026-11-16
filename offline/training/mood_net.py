import torch
from torch import nn


class MultitaskMLP(nn.Module):
    """Shared trunk with a valence/arousal regression head and a quadrant classification head."""

    def __init__(self, in_dim: int = 768, hidden: int = 256, trunk_out: int = 128,
                 n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, trunk_out), nn.ReLU(), nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(trunk_out, 2)
        self.cls_head = nn.Linear(trunk_out, n_classes)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return torch.sigmoid(self.reg_head(h)), self.cls_head(h)