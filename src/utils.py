from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import config


def configure_console_encoding() -> None:
    """Cấu hình console Windows để in log tiếng Việt ổn định hơn."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def log_message(message: str) -> None:
    """In thông báo tiếng Việt ngắn gọn."""
    configure_console_encoding()
    print(f"[INFO] {message}", flush=True)


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    """Cố định seed để kết quả có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directories() -> None:
    """Tạo các thư mục dữ liệu và output cần thiết."""
    for directory in config.OUTPUT_DIRS.values():
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Trả về thiết bị huấn luyện, vẫn chạy tốt trên CPU."""
    return torch.device(config.DEVICE)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Lưu dictionary thành JSON UTF-8."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Đọc JSON UTF-8."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
