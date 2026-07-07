from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.utils import get_device, log_message, set_seed


@dataclass
class TrainingResult:
    """Thông tin sau huấn luyện một mô hình."""

    history: dict[str, list[float]]
    best_valid_loss: float
    best_epoch: int
    training_time_seconds: float


class EarlyStopping:
    """Dừng sớm khi validation loss không cải thiện."""

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, valid_loss: float) -> bool:
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _mean_epoch_loss(total_loss: float, total_samples: int) -> float:
    if total_samples == 0:
        raise ValueError("DataLoader không có mẫu dữ liệu.")
    return total_loss / total_samples


def _estimate_target_mean(train_loader: DataLoader) -> float:
    total_target = 0.0
    total_samples = 0
    for _, targets in train_loader:
        total_target += targets.sum().item()
        total_samples += targets.numel()
    if total_samples == 0:
        raise ValueError("Không thể ước lượng target mean vì train_loader rỗng.")
    return total_target / total_samples


def _initialize_output_bias(model: nn.Module, bias_value: float) -> None:
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear) and module.out_features == 1]
    if not linear_layers:
        return
    output_layer = linear_layers[-1]
    if output_layer.bias is not None:
        with torch.no_grad():
            output_layer.bias.fill_(bias_value)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train một epoch và trả về MSE loss trung bình trên log-price."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return _mean_epoch_loss(total_loss, total_samples)


def validate_model(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Tính validation loss trung bình trên log-price."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in valid_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return _mean_epoch_loss(total_loss, total_samples)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Lưu checkpoint PyTorch kèm metadata để inference."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load state_dict vào model và trả metadata checkpoint."""
    selected_device = device or get_device()
    checkpoint = torch.load(path, map_location=selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model_name: str,
    checkpoint_path: str | Path,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    device: torch.device | None = None,
    model_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """Huấn luyện một mô hình PyTorch bằng pipeline chung."""
    set_seed(config.RANDOM_STATE)
    selected_device = device or get_device()
    model.to(selected_device)
    _initialize_output_bias(model, _estimate_target_mean(train_loader))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    early_stopping = EarlyStopping(patience=patience)
    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    log_message(f"Bắt đầu huấn luyện {model_name} trong tối đa {epochs} epoch.")
    start_time = time.perf_counter()

    progress_bar = tqdm(range(1, epochs + 1), desc=f"Train {model_name}", leave=False)
    for epoch in progress_bar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, selected_device)
        valid_loss = validate_model(model, valid_loader, criterion, selected_device)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        progress_bar.set_postfix({"train_loss": f"{train_loss:.5f}", "valid_loss": f"{valid_loss:.5f}"})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model,
                checkpoint_path,
                {
                    "model_name": model_name,
                    "input_dim": getattr(model, "input_dim", None),
                    "model_params": model_params or {},
                    "history": copy.deepcopy(history),
                    "best_valid_loss": best_valid_loss,
                    "best_epoch": best_epoch,
                },
            )

        if early_stopping.step(valid_loss):
            log_message(f"Dừng sớm {model_name} tại epoch {epoch}.")
            break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state_dict)
    save_checkpoint(
        model,
        checkpoint_path,
        {
            "model_name": model_name,
            "input_dim": getattr(model, "input_dim", None),
            "model_params": model_params or {},
            "history": history,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time,
        },
    )
    log_message(
        f"Huấn luyện xong {model_name}: best_valid_loss={best_valid_loss:.6f}, "
        f"best_epoch={best_epoch}, thời gian={training_time:.2f}s."
    )
    return TrainingResult(
        history=history,
        best_valid_loss=best_valid_loss,
        best_epoch=best_epoch,
        training_time_seconds=training_time,
    )
