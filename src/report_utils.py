from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.models import MODEL_DISPLAY_NAMES
from src.utils import log_message


def _best_by_metric(comparison_df: pd.DataFrame, metric: str, higher_is_better: bool = False) -> str:
    sorted_df = comparison_df.sort_values(metric, ascending=not higher_is_better)
    return str(sorted_df.iloc[0]["model_name"])


def _compare_adaptive(comparison_df: pd.DataFrame, reference_model: str) -> str:
    adaptive = comparison_df.loc[comparison_df["model_name"] == "adaptive_ann"].iloc[0]
    reference = comparison_df.loc[comparison_df["model_name"] == reference_model].iloc[0]
    adaptive_better = adaptive["MAE"] < reference["MAE"] and adaptive["RMSE"] < reference["RMSE"]
    reference_name = MODEL_DISPLAY_NAMES.get(reference_model, reference_model)
    if adaptive_better:
        return (
            f"ANN chuẩn hóa đặc trưng thích nghi có MAE và RMSE thấp hơn {reference_name}, "
            "cho thấy cải thiện trên tập test nội bộ."
        )
    return (
        f"ANN chuẩn hóa đặc trưng thích nghi chưa vượt {reference_name} theo đồng thời MAE và RMSE; "
        "không nên kết luận mô hình đề xuất tốt hơn ở cấu hình thực nghiệm hiện tại."
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_experiment_summary(
    comparison_df: pd.DataFrame | None = None,
    output_path: str | Path = config.EXPERIMENT_SUMMARY_PATH,
    selection_metric: str = "MAE",
) -> Path:
    """Sinh file tóm tắt thực nghiệm bằng tiếng Việt."""
    if comparison_df is None:
        if not config.MODEL_COMPARISON_PATH.exists():
            raise FileNotFoundError("Chưa có outputs/results/model_comparison.csv để tạo báo cáo.")
        comparison_df = pd.read_csv(config.MODEL_COMPARISON_PATH)

    best_mae = _best_by_metric(comparison_df, "MAE")
    best_mse = _best_by_metric(comparison_df, "MSE")
    best_rmse = _best_by_metric(comparison_df, "RMSE")
    best_r2 = _best_by_metric(comparison_df, "R2", higher_is_better=True)
    best_selection = _best_by_metric(comparison_df, selection_metric)

    figure_files = [
        "outputs/figures/loss_ann_baseline.png",
        "outputs/figures/loss_ft_transformer_lite.png",
        "outputs/figures/loss_adaptive_ann.png",
        "outputs/figures/metrics_comparison.png",
        "outputs/figures/actual_vs_predicted_best_model.png",
        "outputs/figures/residual_plot_best_model.png",
    ]

    markdown = f"""# Tóm tắt thực nghiệm

## 1. Tên đề tài

Xây dựng hệ thống dự đoán giá nhà bằng mạng nơ-ron nhân tạo trên bộ dữ liệu Kaggle House Prices.

## 2. Mục tiêu

Dự đoán `SalePrice` của nhà ở dựa trên các đặc trưng về diện tích, số phòng, vị trí, năm xây dựng và các thuộc tính liên quan. Project so sánh ba mô hình deep learning: ANN baseline, FT-Transformer Lite và ANN chuẩn hóa đặc trưng thích nghi do nhóm đề xuất.

## 3. Dataset sử dụng

Bộ dữ liệu chính là `data/raw/train.csv` từ Kaggle House Prices - Advanced Regression Techniques. File `test.csv` không có `SalePrice` nên không dùng để tính MSE/MAE; project tự chia `train.csv` thành train/validation/test nội bộ.

## 4. Các bước tiền xử lý

- Chia dữ liệu trước khi fit scaler/encoder để tránh data leakage.
- Numeric features: điền missing bằng median và chuẩn hóa bằng StandardScaler.
- Categorical features: điền missing bằng `Unknown` và mã hóa bằng OneHotEncoder với `handle_unknown="ignore"`.
- Target `SalePrice` được biến đổi bằng `np.log1p` khi train; metrics được tính trên giá gốc sau `np.expm1`.

## 5. Các đặc trưng mới

- `HouseAge = YrSold - YearBuilt`.
- `RemodAge = YrSold - YearRemodAdd`.
- `TotalArea = GrLivArea + TotalBsmtSF`, phản ánh tổng diện tích sử dụng chính và tránh đếm trùng diện tích tầng.
- `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`.
- `AreaPerRoom = GrLivArea / max(TotRmsAbvGrd, 1)`.
- `HasGarage`, `HasBasement`.
- `Area_Location_Interaction = GrLivArea * NeighborhoodPriceLevel`, trong đó `NeighborhoodPriceLevel` là median `SalePrice` theo `Neighborhood` chỉ fit trên tập train.

## 6. Mô tả mô hình

- **ANN baseline**: MLP nhiều tầng dùng Linear, ReLU, BatchNorm và Dropout.
- **FT-Transformer Lite**: mô hình Transformer gọn cho dữ liệu bảng, xem mỗi feature sau preprocessing như một token scalar rồi học attention giữa các đặc trưng.
- **ANN chuẩn hóa đặc trưng thích nghi**: mô hình đề xuất gồm AdaptiveFeatureNormalization với scale/shift học được, FeatureGate sigmoid theo từng đặc trưng và MLP hồi quy.

## 7. Bảng kết quả metrics

{_dataframe_to_markdown(comparison_df)}

## 8. Nhận xét trung thực

- Model có MAE thấp nhất: **{MODEL_DISPLAY_NAMES.get(best_mae, best_mae)}**.
- Model có MSE thấp nhất: **{MODEL_DISPLAY_NAMES.get(best_mse, best_mse)}**.
- Model có RMSE thấp nhất: **{MODEL_DISPLAY_NAMES.get(best_rmse, best_rmse)}**.
- Model có R2 cao nhất: **{MODEL_DISPLAY_NAMES.get(best_r2, best_r2)}**.
- Best model theo tiêu chí chọn `{selection_metric}`: **{MODEL_DISPLAY_NAMES.get(best_selection, best_selection)}**.
- {_compare_adaptive(comparison_df, "ann_baseline")}
- {_compare_adaptive(comparison_df, "ft_transformer_lite")}
- Kết quả có thể chịu ảnh hưởng bởi kích thước dữ liệu, cách split nội bộ, seed, số epoch và early stopping. Không nên khẳng định mô hình cải tiến vượt trội nếu metrics không thể hiện rõ.

## 9. Danh sách biểu đồ đã xuất

{chr(10).join(f"- `{path}`" for path in figure_files)}

## 10. Gợi ý nội dung đưa vào Word/PPT

- Sơ đồ pipeline: raw data -> split -> feature engineering -> preprocessing -> training -> evaluation.
- Bảng so sánh metrics và thời gian xử lý.
- Biểu đồ loss của từng mô hình.
- Biểu đồ so sánh metrics giữa ba mô hình.
- Biểu đồ giá thực tế so với giá dự đoán và residual plot của model tốt nhất.

## 11. Hạn chế và hướng phát triển

- Dataset có kích thước vừa phải, nên các mô hình deep learning phức tạp có thể chưa phát huy lợi thế rõ rệt.
- Chưa tối ưu hyperparameter sâu cho từng mô hình.
- Có thể mở rộng bằng cross-validation, tuning kiến trúc, regularization, embedding categorical gốc hoặc thêm dữ liệu theo vị trí.
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(markdown, encoding="utf-8")
    log_message(f"Đã tạo báo cáo tóm tắt: {output_path}.")
    return Path(output_path)
