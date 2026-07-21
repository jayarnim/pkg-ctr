import torch
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    def __init__(
        self,
        X,
        y,
    ):
        self.X = torch.tensor(
            data=X.values,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            data=y.values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]