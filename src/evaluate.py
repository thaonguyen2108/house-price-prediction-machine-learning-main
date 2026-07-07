from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.utils import get_device, log_message


@dataclass
class EvaluationResult:
    """Kết quả đánh giá một model trên tập test."""

    metrics: dict[str, float | str]
    predictions: np.ndarray
    actual: np.ndarray


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Dự đoán log-price và đo inference time."""
    selected_device = device or get_device()
    model.to(selected_device)
    model.eval()
    predictions: list[np.ndarray] = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(selected_device)
            batch_predictions = model(features).cpu().numpy().reshape(-1)
            predictions.append(batch_predictions)
    inference_time = time.perf_counter() - start_time

    return np.concatenate(predictions), inference_time


def calculate_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Tính metrics trên giá gốc sau inverse log-transform."""
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, predicted)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    y_true_original: np.ndarray,
    model_name: str,
    training_time_seconds: float,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Đánh giá một model và trả metrics trên đơn vị giá gốc."""
    log_predictions, inference_time = predict(model, data_loader, device=device)
    predicted_prices = np.expm1(log_predictions)
    actual_prices = y_true_original.astype(float)
    metrics = calculate_regression_metrics(actual_prices, predicted_prices)
    metrics.update(
        {
            "model_name": model_name,
            "training_time_seconds": float(training_time_seconds),
            "inference_time_seconds": float(inference_time),
        }
    )
    log_message(
        f"Đánh giá {model_name}: MAE={metrics['MAE']:.2f}, "
        f"RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}."
    )
    return EvaluationResult(metrics=metrics, predictions=predicted_prices, actual=actual_prices)


def save_evaluation_outputs(
    evaluation_results: dict[str, EvaluationResult],
    selection_metric: str = "MAE",
    comparison_path: str | Path = config.MODEL_COMPARISON_PATH,
    predictions_path: str | Path = config.BEST_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, str]:
    """Lưu bảng so sánh và prediction của model tốt nhất."""
    if selection_metric not in {"MAE", "RMSE"}:
        raise ValueError("selection_metric chỉ hỗ trợ 'MAE' hoặc 'RMSE'.")

    comparison_df = pd.DataFrame([result.metrics for result in evaluation_results.values()])
    comparison_df = comparison_df[
        [
            "model_name",
            "MSE",
            "MAE",
            "RMSE",
            "R2",
            "training_time_seconds",
            "inference_time_seconds",
        ]
    ].sort_values(selection_metric, ascending=True)

    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_result = evaluation_results[best_model_name]
    predictions_df = pd.DataFrame(
        {
            "actual_price": best_result.actual,
            "predicted_price": best_result.predictions,
            "residual": best_result.actual - best_result.predictions,
            "model_name": best_model_name,
        }
    )

    Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    log_message(
        f"Đã lưu model_comparison.csv và predictions_best_model.csv. "
        f"Best model theo {selection_metric}: {best_model_name}."
    )
    return comparison_df, best_model_name
