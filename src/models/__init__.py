from __future__ import annotations

from src.models.adaptive_ann import AdaptiveANNRegressor
from src.models.ann_baseline import ANNBaselineRegressor
from src.models.ft_transformer_lite import FTTransformerLiteRegressor

MODEL_DISPLAY_NAMES = {
    "ann_baseline": "ANN baseline",
    "ft_transformer_lite": "FT-Transformer Lite",
    "adaptive_ann": "ANN chuẩn hóa đặc trưng thích nghi",
}

MODEL_DEFAULT_PARAMS = {
    "ann_baseline": {"dropout": 0.2},
    "ft_transformer_lite": {
        "embedding_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.1,
    },
    "adaptive_ann": {"dropout": 0.2},
}


def create_model(model_name: str, input_dim: int, model_params: dict | None = None):
    """Khởi tạo model theo tên chuẩn trong project."""
    params = MODEL_DEFAULT_PARAMS.get(model_name, {}).copy()
    if model_params:
        params.update(model_params)

    if model_name == "ann_baseline":
        return ANNBaselineRegressor(input_dim=input_dim, **params)
    if model_name == "ft_transformer_lite":
        return FTTransformerLiteRegressor(input_dim=input_dim, **params)
    if model_name == "adaptive_ann":
        return AdaptiveANNRegressor(input_dim=input_dim, **params)
    raise ValueError(f"Không hỗ trợ model_name='{model_name}'.")


__all__ = [
    "ANNBaselineRegressor",
    "FTTransformerLiteRegressor",
    "AdaptiveANNRegressor",
    "MODEL_DISPLAY_NAMES",
    "MODEL_DEFAULT_PARAMS",
    "create_model",
]
