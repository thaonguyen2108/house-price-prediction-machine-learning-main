from pathlib import Path

import torch


RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15

TARGET_COLUMN = "SalePrice"
DATA_PATH = "data/raw/train.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
REPORT_DIR = OUTPUT_DIR / "reports"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

OUTPUT_DIRS = {
    "data": DATA_DIR,
    "raw_data": RAW_DATA_DIR,
    "processed_data": PROCESSED_DATA_DIR,
    "outputs": OUTPUT_DIR,
    "models": MODEL_DIR,
    "figures": FIGURE_DIR,
    "results": RESULT_DIR,
    "reports": REPORT_DIR,
    "notebooks": NOTEBOOK_DIR,
}

MODEL_PATHS = {
    "ann_baseline": MODEL_DIR / "ann_baseline.pt",
    "ft_transformer_lite": MODEL_DIR / "ft_transformer_lite.pt",
    "adaptive_ann": MODEL_DIR / "adaptive_ann.pt",
    "best_model": MODEL_DIR / "best_model.pt",
}

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
PREPROCESSING_METADATA_PATH = MODEL_DIR / "preprocessing_metadata.joblib"

MODEL_COMPARISON_PATH = RESULT_DIR / "model_comparison.csv"
BEST_PREDICTIONS_PATH = RESULT_DIR / "predictions_best_model.csv"
EXPERIMENT_SUMMARY_PATH = REPORT_DIR / "experiment_summary.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
