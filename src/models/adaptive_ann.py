from __future__ import annotations

import torch
from torch import nn


class AdaptiveFeatureNormalization(nn.Module):
    """Chuẩn hóa đặc trưng có scale/shift học được cho từng input feature."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm(x)
        return normalized * self.scale + self.shift


class FeatureGate(nn.Module):
    """Học trọng số sigmoid để điều chỉnh mức ảnh hưởng của từng đặc trưng."""

    def __init__(self, input_dim: int, initial_logit: float = 2.0) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((input_dim,), initial_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logits)
        return x * gate


class AdaptiveANNRegressor(nn.Module):
    """ANN cải tiến: AdaptiveFeatureNormalization -> FeatureGate -> MLP."""

    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.adaptive_norm = AdaptiveFeatureNormalization(input_dim)
        self.feature_gate = FeatureGate(input_dim)
        self.regressor = nn.Sequential(
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
        normalized = self.adaptive_norm(x)
        gated = self.feature_gate(normalized)
        return self.regressor(gated)
