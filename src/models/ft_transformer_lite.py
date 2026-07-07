from __future__ import annotations

import torch
from torch import nn


class FTTransformerLiteRegressor(nn.Module):
    """FT-Transformer Lite cho dữ liệu bảng đã preprocess.

    Mỗi đặc trưng scalar được xem như một token, sau đó chiếu sang không gian
    embedding và đưa qua TransformerEncoder gọn để chạy được trên CPU.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim phải chia hết cho num_heads.")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.value_projection = nn.Linear(1, embedding_dim)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        encoded_tokens = self.encoder(tokens)
        pooled = encoded_tokens.mean(dim=1)
        return self.regression_head(pooled)
