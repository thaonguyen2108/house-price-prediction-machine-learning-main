from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data_loader import load_raw_data, validate_required_columns
from src.feature_engineering import (
    NeighborhoodEncoding,
    build_features,
    fit_neighborhood_encoding,
)
from src.utils import ensure_directories, log_message


@dataclass
class PreprocessingResult:
    """Kết quả tiền xử lý dùng cho huấn luyện và demo."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train_log: np.ndarray
    y_valid_log: np.ndarray
    y_test_log: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    metadata: dict[str, Any]


def _split_train_valid_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    valid_relative_size = config.VALID_SIZE / (1.0 - config.TEST_SIZE)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_relative_size,
        random_state=config.RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[config.TARGET_COLUMN].copy()
    X = df.drop(columns=[config.TARGET_COLUMN, "Id"], errors="ignore").copy()
    return X, y


def _detect_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in df.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _to_feature_frame(preprocessor: ColumnTransformer, values: np.ndarray) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(values, columns=feature_names)


def _save_processed_files(result: PreprocessingResult) -> None:
    processed_dir = config.PROCESSED_DATA_DIR
    result.X_train.to_csv(processed_dir / "X_train.csv", index=False)
    result.X_valid.to_csv(processed_dir / "X_valid.csv", index=False)
    result.X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame(
        {
            config.TARGET_COLUMN: result.y_train,
            f"log_{config.TARGET_COLUMN}": result.y_train_log,
        }
    ).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame(
        {
            config.TARGET_COLUMN: result.y_valid,
            f"log_{config.TARGET_COLUMN}": result.y_valid_log,
        }
    ).to_csv(processed_dir / "y_valid.csv", index=False)
    pd.DataFrame(
        {
            config.TARGET_COLUMN: result.y_test,
            f"log_{config.TARGET_COLUMN}": result.y_test_log,
        }
    ).to_csv(processed_dir / "y_test.csv", index=False)


def _get_raw_default_values(X_train_raw: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in X_train_raw.columns:
        series = X_train_raw[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            mode = series.dropna().mode()
            defaults[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    return defaults


def _get_raw_numeric_ranges(X_train_raw: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for column in X_train_raw.select_dtypes(include=["number", "bool"]).columns:
        series = X_train_raw[column].dropna()
        if not series.empty:
            ranges[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return ranges


def prepare_datasets(
    data_path: str | Path = config.DATA_PATH,
    save_outputs: bool = True,
) -> PreprocessingResult:
    """Chia dữ liệu, tạo đặc trưng, fit preprocessing và lưu artifact."""
    ensure_directories()
    raw_df = load_raw_data(data_path)
    validate_required_columns(raw_df)

    train_df, valid_df, test_df = _split_train_valid_test(raw_df)
    X_train_raw, y_train = _split_features_target(train_df)
    X_valid_raw, y_valid = _split_features_target(valid_df)
    X_test_raw, y_test = _split_features_target(test_df)

    neighborhood_encoding = fit_neighborhood_encoding(X_train_raw, y_train)
    X_train_features = build_features(X_train_raw, neighborhood_encoding)
    X_valid_features = build_features(X_valid_raw, neighborhood_encoding)
    X_test_features = build_features(X_test_raw, neighborhood_encoding)

    numeric_features, categorical_features = _detect_feature_types(X_train_features)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    X_train_array = preprocessor.fit_transform(X_train_features)
    X_valid_array = preprocessor.transform(X_valid_features)
    X_test_array = preprocessor.transform(X_test_features)

    if np.isnan(X_train_array).any() or np.isnan(X_valid_array).any() or np.isnan(X_test_array).any():
        raise ValueError("Dữ liệu sau preprocessing vẫn còn NaN. Cần kiểm tra lại pipeline.")

    X_train_processed = _to_feature_frame(preprocessor, X_train_array)
    X_valid_processed = _to_feature_frame(preprocessor, X_valid_array)
    X_test_processed = _to_feature_frame(preprocessor, X_test_array)

    y_train_values = y_train.to_numpy(dtype=np.float32)
    y_valid_values = y_valid.to_numpy(dtype=np.float32)
    y_test_values = y_test.to_numpy(dtype=np.float32)
    y_train_log = np.log1p(y_train_values).astype(np.float32)
    y_valid_log = np.log1p(y_valid_values).astype(np.float32)
    y_test_log = np.log1p(y_test_values).astype(np.float32)

    metadata = {
        "raw_input_columns": X_train_raw.columns.tolist(),
        "engineered_feature_columns": X_train_features.columns.tolist(),
        "processed_feature_names": X_train_processed.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "raw_default_values": _get_raw_default_values(X_train_raw),
        "raw_numeric_ranges": _get_raw_numeric_ranges(X_train_raw),
        "neighborhood_price_levels": neighborhood_encoding.price_levels,
        "global_median_price": neighborhood_encoding.global_median_price,
        "target_column": config.TARGET_COLUMN,
        "input_dim": int(X_train_processed.shape[1]),
        "train_shape": list(X_train_processed.shape),
        "valid_shape": list(X_valid_processed.shape),
        "test_shape": list(X_test_processed.shape),
    }

    result = PreprocessingResult(
        X_train=X_train_processed,
        X_valid=X_valid_processed,
        X_test=X_test_processed,
        y_train_log=y_train_log,
        y_valid_log=y_valid_log,
        y_test_log=y_test_log,
        y_train=y_train_values,
        y_valid=y_valid_values,
        y_test=y_test_values,
        preprocessor=preprocessor,
        metadata=metadata,
    )

    if save_outputs:
        joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
        joblib.dump(metadata, config.PREPROCESSING_METADATA_PATH)
        _save_processed_files(result)

    log_message(
        "Preprocessing hoàn tất: "
        f"train={result.X_train.shape}, valid={result.X_valid.shape}, test={result.X_test.shape}."
    )
    return result


def load_preprocessor() -> ColumnTransformer:
    """Đọc preprocessor đã fit."""
    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Chưa có outputs/models/preprocessor.joblib. Hãy chạy preprocessing trước.")
    return joblib.load(config.PREPROCESSOR_PATH)


def load_preprocessing_metadata() -> dict[str, Any]:
    """Đọc metadata phục vụ inference trong Streamlit."""
    if not config.PREPROCESSING_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Chưa có outputs/models/preprocessing_metadata.joblib. Hãy chạy pipeline train trước."
        )
    return joblib.load(config.PREPROCESSING_METADATA_PATH)


def transform_raw_input(raw_input: pd.DataFrame, preprocessor: ColumnTransformer, metadata: dict[str, Any]) -> pd.DataFrame:
    """Tạo feature engineering và transform một batch dữ liệu raw bằng preprocessor đã fit."""
    encoding = NeighborhoodEncoding(
        price_levels=metadata["neighborhood_price_levels"],
        global_median_price=float(metadata["global_median_price"]),
    )
    features = build_features(raw_input, encoding)
    transformed = preprocessor.transform(features)
    return pd.DataFrame(transformed, columns=metadata["processed_feature_names"])
