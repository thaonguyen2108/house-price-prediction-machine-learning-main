from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.models import MODEL_DISPLAY_NAMES
from src.utils import log_message


plt.rcParams["font.family"] = "DejaVu Sans"


def _display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def plot_loss_curve(history: dict[str, list[float]], model_name: str, output_path: str | Path) -> None:
    """Vẽ loss train/validation của một mô hình."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("train_loss", []), label="Train loss")
    plt.plot(history.get("valid_loss", []), label="Validation loss")
    plt.title(f"Đường loss - {_display_name(model_name)}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE trên log-price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_all_loss_curves(histories: dict[str, dict[str, list[float]]]) -> None:
    """Lưu biểu đồ loss riêng cho từng mô hình."""
    output_names = {
        "ann_baseline": "loss_ann_baseline.png",
        "ft_transformer_lite": "loss_ft_transformer_lite.png",
        "adaptive_ann": "loss_adaptive_ann.png",
    }
    for model_name, history in histories.items():
        output_path = config.FIGURE_DIR / output_names.get(model_name, f"loss_{model_name}.png")
        plot_loss_curve(history, model_name, output_path)


def plot_metrics_comparison(comparison_df: pd.DataFrame, output_path: str | Path = config.FIGURE_DIR / "metrics_comparison.png") -> None:
    """Vẽ biểu đồ so sánh MSE, MAE, RMSE, R2."""
    metrics = ["MSE", "MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    labels = [_display_name(name) for name in comparison_df["model_name"]]

    for axis, metric in zip(axes, metrics):
        axis.bar(labels, comparison_df[metric])
        axis.set_title(f"So sánh {metric}")
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    output_path: str | Path = config.FIGURE_DIR / "actual_vs_predicted_best_model.png",
) -> None:
    """Vẽ giá thực tế so với giá dự đoán của model tốt nhất."""
    actual = predictions_df["actual_price"]
    predicted = predictions_df["predicted_price"]
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.65)
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", label="Dự đoán lý tưởng")
    plt.title("Giá thực tế và giá dự đoán - model tốt nhất")
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_residuals(
    predictions_df: pd.DataFrame,
    output_path: str | Path = config.FIGURE_DIR / "residual_plot_best_model.png",
) -> None:
    """Vẽ residual của model tốt nhất."""
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions_df["predicted_price"], predictions_df["residual"], alpha=0.65)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual plot - model tốt nhất")
    plt.xlabel("Giá dự đoán")
    plt.ylabel("Residual = giá thực tế - giá dự đoán")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_all_figures(histories: dict[str, dict[str, list[float]]], comparison_df: pd.DataFrame) -> None:
    """Tạo toàn bộ biểu đồ phục vụ Word/PPT."""
    plot_all_loss_curves(histories)
    plot_metrics_comparison(comparison_df)
    predictions_df = pd.read_csv(config.BEST_PREDICTIONS_PATH)
    plot_actual_vs_predicted(predictions_df)
    plot_residuals(predictions_df)
    log_message("Đã lưu toàn bộ biểu đồ vào outputs/figures.")
