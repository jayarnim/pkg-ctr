from .split import stratified_split
from .scaler import scaler
from ..dataloader import build_dataloader
from ..dataloader import DataLoader
from ..config.config.datamodule import DataModuleCfg
import pandas as pd


def build_datamodule(
    X: pd.DataFrame, 
    y: pd.Series,
    cfg: DataModuleCfg, 
    scaling: list=None,
) -> dict[str, DataLoader]:
    split = stratified_split(
        X=X,
        y=y,
        **vars(cfg.split),
    )

    if scaling is not None:
        split = scaler(
            split=split,
            cols=scaling,
        )

    return {
        k: build_dataloader(
            *v,
            cfg=cfg.dataloader,
        )
        for k, v in split.items()
    }