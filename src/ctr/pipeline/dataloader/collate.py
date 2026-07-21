import torch


def collate_fn(batch):
    X_batch, y_batch = zip(*batch)

    X_batch = torch.stack(
        tensors=X_batch,
        dim=0,
    )

    y_batch = torch.stack(
        tensors=y_batch,
        dim=0,
    )

    return X_batch, y_batch