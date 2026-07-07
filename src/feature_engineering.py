from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


REQUIRED_ENGINEERING_COLUMNS = [
    "YrSold",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "TotRmsAbvGrd",
    "GarageArea",
    "Neighborhood",
]


@dataclass(frozen=True)
class NeighborhoodEncoding:
    """Bảng mã hóa mức giá khu vực được fit chỉ trên tập train."""

    price_levels: dict[str, float]
    global_median_price: float


def _check_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_ENGINEERING_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Thiếu các cột cần cho feature engineering: {missing_columns}")


def add_basic_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng phi mục tiêu từ dữ liệu nhà ở."""
    _check_required_columns(df)
    engineered = df.copy()

    engineered["HouseAge"] = engineered["YrSold"] - engineered["YearBuilt"]
    engineered["RemodAge"] = engineered["YrSold"] - engineered["YearRemodAdd"]
    engineered["TotalArea"] = (
        engineered["GrLivArea"]
        + engineered["TotalBsmtSF"]
    )
    engineered["TotalBath"] = (
        engineered["FullBath"]
        + 0.5 * engineered["HalfBath"]
        + engineered["BsmtFullBath"]
        + 0.5 * engineered["BsmtHalfBath"]
    )
    rooms = engineered["TotRmsAbvGrd"].clip(lower=1)
    engineered["AreaPerRoom"] = engineered["GrLivArea"] / rooms
    engineered["HasGarage"] = (engineered["GarageArea"] > 0).astype(int)
    engineered["HasBasement"] = (engineered["TotalBsmtSF"] > 0).astype(int)
    return engineered


def fit_neighborhood_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_column: str = config.TARGET_COLUMN,
) -> NeighborhoodEncoding:
    """Fit median SalePrice theo Neighborhood chỉ từ tập train."""
    if "Neighborhood" not in X_train.columns:
        raise ValueError("Thiếu cột Neighborhood để tạo Area_Location_Interaction.")
    train_with_target = X_train[["Neighborhood"]].copy()
    train_with_target[target_column] = y_train.to_numpy()
    levels = (
        train_with_target.groupby("Neighborhood")[target_column]
        .median()
        .astype(float)
        .to_dict()
    )
    global_median = float(np.median(y_train.to_numpy()))
    return NeighborhoodEncoding(price_levels=levels, global_median_price=global_median)


def apply_neighborhood_encoding(
    df: pd.DataFrame,
    encoding: NeighborhoodEncoding,
) -> pd.DataFrame:
    """Áp dụng target-based encoding đã fit trên train cho một split dữ liệu."""
    encoded = df.copy()
    encoded["NeighborhoodPriceLevel"] = (
        encoded["Neighborhood"]
        .map(encoding.price_levels)
        .fillna(encoding.global_median_price)
        .astype(float)
    )
    encoded["Area_Location_Interaction"] = (
        encoded["GrLivArea"].astype(float) * encoded["NeighborhoodPriceLevel"]
    )
    return encoded


def build_features(
    df: pd.DataFrame,
    encoding: NeighborhoodEncoding,
) -> pd.DataFrame:
    """Tạo toàn bộ đặc trưng mới cho một split đã chia sẵn."""
    engineered = add_basic_engineered_features(df)
    return apply_neighborhood_encoding(engineered, encoding)
