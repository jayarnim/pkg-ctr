from .split import stratified_split_builder
from .scaler import scaler
from .dataloader.builder import dataloader_builder


def pipeline_builder(
    X, 
    y,
    cfg, 
    scaling: list=None,
):
    split = stratified_split_builder(
        X=X,
        y=y,
        cfg=cfg.split,
    )

    if scaling is not None:
        split = scaler(
            split=split,
            cols=scaling,
        )

    return {
        k: dataloader_builder(
            *v,
            cfg=cfg.dataloader,
        )
        for k, v in split.items()
    }