import pandas as pd
import torch
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame, 
        y: pd.Series,
    ):
        self.X = torch.tensor(
            data=X.values,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            data=y.values,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, 
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]