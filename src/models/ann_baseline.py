from __future__ import annotations

import torch
from torch import nn


class ANNBaselineRegressor(nn.Module):
    """ANN baseline cho bài toán hồi quy giá nhà."""

    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
