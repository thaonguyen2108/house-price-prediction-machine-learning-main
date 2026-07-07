from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.datasets import TabularRegressionDataset
from src.evaluate import evaluate_model, save_evaluation_outputs
from src.models import MODEL_DEFAULT_PARAMS, create_model
from src.preprocessing import PreprocessingResult, prepare_datasets
from src.report_utils import generate_experiment_summary
from src.training import TrainingResult, train_model
from src.utils import configure_console_encoding, ensure_directories, get_device, log_message, set_seed
from src.visualize import create_all_figures


MODEL_ORDER = ["ann_baseline", "ft_transformer_lite", "adaptive_ann"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline dự đoán giá nhà bằng PyTorch.")
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def create_data_loaders(data: PreprocessingResult, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        TabularRegressionDataset(data.X_train, data.y_train_log),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        TabularRegressionDataset(data.X_valid, data.y_valid_log),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TabularRegressionDataset(data.X_test, data.y_test_log),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


def train_all_models(
    data: PreprocessingResult,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
) -> tuple[dict[str, torch.nn.Module], dict[str, TrainingResult]]:
    models: dict[str, torch.nn.Module] = {}
    training_results: dict[str, TrainingResult] = {}
    device = get_device()

    for model_name in MODEL_ORDER:
        model = create_model(
            model_name,
            input_dim=data.X_train.shape[1],
            model_params=MODEL_DEFAULT_PARAMS[model_name],
        )
        result = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            model_name=model_name,
            checkpoint_path=config.MODEL_PATHS[model_name],
            epochs=epochs,
            device=device,
            model_params=MODEL_DEFAULT_PARAMS[model_name],
        )
        models[model_name] = model
        training_results[model_name] = result

    return models, training_results


def load_trained_models(input_dim: int) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, list[float]]], dict[str, float]]:
    models: dict[str, torch.nn.Module] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    training_times: dict[str, float] = {}
    device = get_device()

    for model_name in MODEL_ORDER:
        checkpoint_path = config.MODEL_PATHS[model_name]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Chưa có checkpoint {checkpoint_path}. Hãy chạy --mode train hoặc --mode all trước.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = create_model(
            model_name,
            input_dim=int(checkpoint.get("input_dim") or input_dim),
            model_params=checkpoint.get("model_params") or MODEL_DEFAULT_PARAMS[model_name],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models[model_name] = model
        histories[model_name] = checkpoint.get("history", {"train_loss": [], "valid_loss": []})
        training_times[model_name] = float(checkpoint.get("training_time_seconds", 0.0))

    return models, histories, training_times


def evaluate_and_report(
    data: PreprocessingResult,
    test_loader: DataLoader,
    models: dict[str, torch.nn.Module],
    histories: dict[str, dict[str, list[float]]],
    training_times: dict[str, float],
) -> pd.DataFrame:
    evaluation_results = {}
    for model_name, model in models.items():
        evaluation_results[model_name] = evaluate_model(
            model=model,
            data_loader=test_loader,
            y_true_original=data.y_test,
            model_name=model_name,
            training_time_seconds=training_times[model_name],
            device=get_device(),
        )

    comparison_df, best_model_name = save_evaluation_outputs(evaluation_results, selection_metric="MAE")
    source_checkpoint = config.MODEL_PATHS[best_model_name]
    shutil.copyfile(source_checkpoint, config.MODEL_PATHS["best_model"])

    create_all_figures(histories, comparison_df)
    generate_experiment_summary(comparison_df, selection_metric="MAE")
    return comparison_df


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    ensure_directories()
    set_seed(config.RANDOM_STATE)

    log_message(f"Chạy pipeline với mode={args.mode}, epochs={args.epochs}, batch_size={args.batch_size}.")
    data = prepare_datasets()
    train_loader, valid_loader, test_loader = create_data_loaders(data, args.batch_size)

    if args.mode == "train":
        train_all_models(data, train_loader, valid_loader, args.epochs)
        log_message("Hoàn tất mode=train. Chạy --mode evaluate hoặc --mode all để tạo metrics/biểu đồ/báo cáo.")
        return

    if args.mode == "evaluate":
        models, histories, training_times = load_trained_models(data.X_train.shape[1])
        evaluate_and_report(data, test_loader, models, histories, training_times)
        log_message("Hoàn tất mode=evaluate.")
        return

    models, training_results = train_all_models(data, train_loader, valid_loader, args.epochs)
    histories = {name: result.history for name, result in training_results.items()}
    training_times = {name: result.training_time_seconds for name, result in training_results.items()}
    evaluate_and_report(data, test_loader, models, histories, training_times)
    log_message("Hoàn tất toàn bộ pipeline.")


if __name__ == "__main__":
    main()
