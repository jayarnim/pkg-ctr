import torch
from torch.utils.data import DataLoader
from .dataset import CTRDataset


def dataloader_generator(
    dataset,
    collate_fn, 
    batch_size, 
    shuffle,
):
    kwargs = dict(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return DataLoader(**kwargs)
