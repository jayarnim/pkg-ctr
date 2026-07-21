from .dataset import CTRDataset
from .collate import collate_fn
from .dataloader import dataloader_generator


def dataloader_builder(X, y, cfg):
    dataset = CTRDataset(
        X=X,
        y=y,
    )

    return dataloader_generator(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )