from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.models import MODEL_DISPLAY_NAMES, create_model
from src.preprocessing import transform_raw_input


st.set_page_config(page_title="Dự đoán giá nhà", layout="wide")

AREA_COLUMNS = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LotArea", "GarageArea"]
YEAR_COLUMNS = ["YearBuilt", "YearRemodAdd", "YrSold"]
OTHER_NUMERIC_COLUMNS = [
    "TotRmsAbvGrd",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
]


@st.cache_resource
def load_artifacts() -> tuple[Any, dict[str, Any], torch.nn.Module, str]:
    if not config.PREPROCESSOR_PATH.exists() or not config.PREPROCESSING_METADATA_PATH.exists():
        raise FileNotFoundError("Chưa có preprocessor/metadata. Hãy chạy pipeline train trước.")
    if not config.MODEL_PATHS["best_model"].exists():
        raise FileNotFoundError("Chưa có best_model.pt. Hãy chạy main.py --mode all trước.")

    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    metadata = joblib.load(config.PREPROCESSING_METADATA_PATH)
    checkpoint = torch.load(config.MODEL_PATHS["best_model"], map_location="cpu")
    model_name = checkpoint["model_name"]
    model = create_model(
        model_name=model_name,
        input_dim=int(checkpoint["input_dim"]),
        model_params=checkpoint.get("model_params"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return preprocessor, metadata, model, model_name


@st.cache_data
def load_train_reference() -> tuple[dict[str, dict[str, float]], list[str]]:
    data_path = config.PROJECT_ROOT / config.DATA_PATH
    if not data_path.exists():
        return {}, []
    train_df = pd.read_csv(data_path)
    ranges: dict[str, dict[str, float]] = {}
    for column in train_df.select_dtypes(include=["number", "bool"]).columns:
        if column in {"Id", config.TARGET_COLUMN}:
            continue
        series = train_df[column].dropna()
        if not series.empty:
            ranges[column] = {"min": float(series.min()), "max": float(series.max())}
    neighborhoods = sorted(train_df["Neighborhood"].dropna().astype(str).unique().tolist())
    return ranges, neighborhoods


def number_default(defaults: dict[str, Any], column: str, fallback: float) -> float:
    return float(defaults.get(column, fallback))


def int_default(defaults: dict[str, Any], column: str, fallback: int) -> int:
    return int(round(number_default(defaults, column, fallback)))


def get_numeric_ranges(metadata: dict[str, Any]) -> dict[str, dict[str, float]]:
    ranges = metadata.get("raw_numeric_ranges")
    if isinstance(ranges, dict) and ranges:
        return ranges
    train_ranges, _ = load_train_reference()
    return train_ranges


def get_neighborhoods(metadata: dict[str, Any]) -> list[str]:
    _, raw_neighborhoods = load_train_reference()
    if raw_neighborhoods:
        return raw_neighborhoods
    return sorted(metadata["neighborhood_price_levels"].keys())


def build_input_row(metadata: dict[str, Any], user_values: dict[str, Any]) -> pd.DataFrame:
    row = metadata["raw_default_values"].copy()
    row.update(user_values)
    return pd.DataFrame([row], columns=metadata["raw_input_columns"])


def predict_price(raw_input: pd.DataFrame, preprocessor: Any, metadata: dict[str, Any], model: torch.nn.Module) -> float:
    processed = transform_raw_input(raw_input, preprocessor, metadata)
    features = torch.as_tensor(processed.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        log_price = model(features).item()
    return float(np.expm1(log_price))


def validate_inputs(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if values["YearBuilt"] > values["YearRemodAdd"]:
        errors.append("Năm xây dựng phải nhỏ hơn hoặc bằng năm sửa chữa/cải tạo.")
    if values["YearRemodAdd"] > values["YrSold"]:
        errors.append("Năm sửa chữa/cải tạo phải nhỏ hơn hoặc bằng năm bán.")
    if values["BedroomAbvGr"] > values["TotRmsAbvGrd"]:
        errors.append("Số phòng ngủ không được lớn hơn tổng số phòng trên mặt đất.")

    for column in AREA_COLUMNS:
        if values[column] < 0:
            errors.append(f"{column} phải lớn hơn hoặc bằng 0.")

    for column in OTHER_NUMERIC_COLUMNS:
        if values[column] < 0:
            errors.append(f"{column} phải lớn hơn hoặc bằng 0.")
    return errors


def build_range_warnings(values: dict[str, Any], ranges: dict[str, dict[str, float]]) -> list[str]:
    warnings: list[str] = []
    year_sold_range = ranges.get("YrSold")

    for column in YEAR_COLUMNS:
        range_info = ranges.get(column)
        if not range_info:
            continue
        value = float(values[column])
        if value < range_info["min"] or value > range_info["max"]:
            if year_sold_range:
                warnings.append(
                    "Lưu ý: Giá trị năm đang nằm ngoài khoảng dữ liệu huấn luyện. "
                    f"Mô hình được huấn luyện chủ yếu trên dữ liệu bán nhà trong giai đoạn "
                    f"{int(year_sold_range['min'])}-{int(year_sold_range['max'])}, "
                    "nên kết quả dự đoán có thể kém chính xác khi áp dụng cho năm ngoài khoảng này."
                )
            else:
                warnings.append(
                    f"{column} nằm ngoài khoảng dữ liệu huấn luyện "
                    f"({range_info['min']:.0f}-{range_info['max']:.0f}), kết quả dự đoán có thể kém chính xác."
                )

    for column in AREA_COLUMNS + OTHER_NUMERIC_COLUMNS:
        range_info = ranges.get(column)
        if not range_info:
            continue
        value = float(values[column])
        if value < range_info["min"] or value > range_info["max"]:
            warnings.append(
                f"{column}: Giá trị này nằm ngoài khoảng dữ liệu huấn luyện "
                f"({range_info['min']:.0f}-{range_info['max']:.0f}), kết quả dự đoán có thể kém tin cậy."
            )

    if values["LotArea"] < values["GrLivArea"]:
        warnings.append(
            "Diện tích đất nhỏ hơn diện tích sinh hoạt trên mặt đất. "
            "Trường hợp này vẫn được dự đoán, nhưng nên kiểm tra lại dữ liệu nhập."
        )

    return list(dict.fromkeys(warnings))


try:
    preprocessor, metadata, model, model_name = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Chạy lệnh: .\\.venv\\Scripts\\python.exe main.py --mode all")
    st.stop()


st.title("Hệ thống dự đoán giá nhà")
st.caption("Demo local bằng PyTorch và Streamlit trên Kaggle House Prices.")

tab_predict, tab_compare, tab_figures, tab_info = st.tabs(
    ["Dự đoán giá nhà", "So sánh mô hình", "Biểu đồ thực nghiệm", "Thông tin mô hình"]
)

with tab_predict:
    st.subheader("Nhập thông tin nhà")
    st.info(
        "Ứng dụng demo được huấn luyện trên bộ Kaggle House Prices/Ames Housing. "
        "Dự đoán đáng tin cậy hơn khi dữ liệu đầu vào nằm gần phạm vi dữ liệu huấn luyện."
    )

    defaults = metadata["raw_default_values"]
    numeric_ranges = get_numeric_ranges(metadata)
    neighborhoods = get_neighborhoods(metadata)
    default_neighborhood = str(defaults.get("Neighborhood", neighborhoods[0]))
    default_neighborhood_index = neighborhoods.index(default_neighborhood) if default_neighborhood in neighborhoods else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        gr_liv_area = st.number_input(
            "Diện tích sinh hoạt trên mặt đất (GrLivArea)",
            value=int_default(defaults, "GrLivArea", 1500),
            step=1,
        )
        total_bsmt_sf = st.number_input(
            "Diện tích tầng hầm (TotalBsmtSF)",
            value=int_default(defaults, "TotalBsmtSF", 800),
            step=1,
        )
        first_flr_sf = st.number_input(
            "Diện tích tầng 1 (1stFlrSF)",
            value=int_default(defaults, "1stFlrSF", 1000),
            step=1,
        )
        second_flr_sf = st.number_input(
            "Diện tích tầng 2 (2ndFlrSF)",
            value=int_default(defaults, "2ndFlrSF", 0),
            step=1,
        )
        lot_area = st.number_input(
            "Diện tích đất (LotArea)",
            value=int_default(defaults, "LotArea", 9000),
            step=1,
        )
    with col2:
        total_rooms = st.number_input(
            "Tổng số phòng trên mặt đất (TotRmsAbvGrd)",
            value=int_default(defaults, "TotRmsAbvGrd", 6),
            step=1,
        )
        bedrooms = st.number_input(
            "Số phòng ngủ (BedroomAbvGr)",
            value=int_default(defaults, "BedroomAbvGr", 3),
            step=1,
        )
        full_bath = st.number_input(
            "Số phòng tắm đầy đủ (FullBath)",
            value=int_default(defaults, "FullBath", 2),
            step=1,
        )
        half_bath = st.number_input(
            "Số phòng tắm phụ (HalfBath)",
            value=int_default(defaults, "HalfBath", 0),
            step=1,
        )
        bsmt_full_bath = st.number_input(
            "Số phòng tắm đầy đủ ở tầng hầm (BsmtFullBath)",
            value=int_default(defaults, "BsmtFullBath", 0),
            step=1,
        )
        bsmt_half_bath = st.number_input(
            "Số phòng tắm phụ ở tầng hầm (BsmtHalfBath)",
            value=int_default(defaults, "BsmtHalfBath", 0),
            step=1,
        )
    with col3:
        year_built = st.number_input(
            "Năm xây dựng (YearBuilt)",
            value=int_default(defaults, "YearBuilt", 1970),
            step=1,
        )
        year_remod = st.number_input(
            "Năm sửa chữa/cải tạo (YearRemodAdd)",
            value=int_default(defaults, "YearRemodAdd", 1990),
            step=1,
        )
        yr_sold = st.number_input(
            "Năm bán (YrSold)",
            value=int_default(defaults, "YrSold", 2010),
            step=1,
        )
        garage_area = st.number_input(
            "Diện tích garage (GarageArea)",
            value=int_default(defaults, "GarageArea", 400),
            step=1,
        )
        neighborhood = st.selectbox("Khu vực (Neighborhood)", neighborhoods, index=default_neighborhood_index)

    user_values = {
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "LotArea": lot_area,
        "TotRmsAbvGrd": total_rooms,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BsmtFullBath": bsmt_full_bath,
        "BsmtHalfBath": bsmt_half_bath,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "YrSold": yr_sold,
        "GarageArea": garage_area,
        "Neighborhood": neighborhood,
    }

    validation_errors = validate_inputs(user_values)
    for warning in build_range_warnings(user_values, numeric_ranges):
        st.warning(warning)

    if validation_errors:
        for error in validation_errors:
            st.error(error)

    if st.button("Dự đoán giá", type="primary", disabled=bool(validation_errors)):
        raw_input = build_input_row(metadata, user_values)
        predicted_price = predict_price(raw_input, preprocessor, metadata, model)
        st.metric("Giá dự đoán", f"${predicted_price:,.0f}")
        st.caption(f"Model đang dùng: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}.")

with tab_compare:
    st.subheader("Bảng so sánh mô hình")
    if config.MODEL_COMPARISON_PATH.exists():
        comparison_df = pd.read_csv(config.MODEL_COMPARISON_PATH)
        st.dataframe(comparison_df, use_container_width=True)
        best_row = comparison_df.sort_values("MAE").iloc[0]
        st.success(
            f"Best model theo MAE: {MODEL_DISPLAY_NAMES.get(best_row['model_name'], best_row['model_name'])} "
            f"với MAE = {best_row['MAE']:,.2f}."
        )
    else:
        st.warning("Chưa có model_comparison.csv. Hãy chạy pipeline trước.")

with tab_figures:
    st.subheader("Biểu đồ thực nghiệm")
    figure_paths = [
        config.FIGURE_DIR / "loss_ann_baseline.png",
        config.FIGURE_DIR / "loss_ft_transformer_lite.png",
        config.FIGURE_DIR / "loss_adaptive_ann.png",
        config.FIGURE_DIR / "metrics_comparison.png",
        config.FIGURE_DIR / "actual_vs_predicted_best_model.png",
        config.FIGURE_DIR / "residual_plot_best_model.png",
    ]
    for figure_path in figure_paths:
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_path.name, use_container_width=True)
        else:
            st.warning(f"Chưa có biểu đồ: {figure_path.name}")

with tab_info:
    st.subheader("Thông tin mô hình")
    st.markdown(
        """
        **ANN baseline**: MLP truyền thống cho dữ liệu bảng sau preprocessing.

        **FT-Transformer Lite**: phiên bản Transformer gọn, xem mỗi feature đã preprocess như một token và học quan hệ giữa các đặc trưng bằng self-attention.

        **ANN chuẩn hóa đặc trưng thích nghi**: mô hình đề xuất gồm lớp chuẩn hóa có `scale/shift` học được, lớp `FeatureGate` học trọng số sigmoid theo từng feature và MLP hồi quy.

        Metrics trong project được tính trên giá gốc sau khi inverse log-transform bằng `np.expm1`.
        """
    )
