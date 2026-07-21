from .dataset import CTRDataset
from .collate import collate_fn
from torch.utils.data import DataLoader
import pandas as pd
from ..config.config.datamodule import DataloaderCfg


def build_dataloader(
    X: pd.DataFrame, 
    y: pd.Series, 
    cfg: DataloaderCfg,
) -> DataLoader:
    dataset = CTRDataset(
        X=X,
        y=y,
    )

    kwargs = dict(
        dataset=dataset,
        collate_fn=collate_fn,
        **vars(cfg),
    )
    return DataLoader(**kwargs)