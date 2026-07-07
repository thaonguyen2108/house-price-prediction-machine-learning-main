from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularRegressionDataset(Dataset):
    """Dataset PyTorch cho hồi quy dữ liệu bảng."""

    def __init__(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> None:
        X_values = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_values = y.to_numpy() if isinstance(y, pd.Series) else y

        self.X = torch.as_tensor(X_values, dtype=torch.float32)
        self.y = torch.as_tensor(y_values, dtype=torch.float32).view(-1, 1)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Số dòng của X và y không khớp.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
