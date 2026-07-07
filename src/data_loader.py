from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_raw_data(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu train.csv của Kaggle House Prices."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu huấn luyện. "
            "Vui lòng đặt file Kaggle House Prices train.csv vào data/raw/train.csv rồi chạy lại."
        )
    return pd.read_csv(data_path)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Kiểm tra cột mục tiêu bắt buộc."""
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Thiếu cột mục tiêu '{config.TARGET_COLUMN}'. "
            "File train.csv phải có SalePrice để train/evaluate."
        )
