from .split import stratified_split_builder
from .encoder import encoder
from .scaler import scaler
from .dataloader.builder import dataloader_builder


def pipeline_builder(
    X, 
    y,
    cfg, 
    encoding: list=None,
    scaling: list=None,
):
    if encoding is not None:
        X = encoder(
            X=X,
            cols=encoding,
        )

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