from typing import Literal
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torchmetrics.classification import BinaryROC


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)
        self.current_epoch = 0

    def __call__(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        self.current_epoch += 1

        # train
        self.model.train()

        # reset epoch loss
        epoch_loss = 0.0

        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {self.current_epoch} TRN"
        )
        for X, y in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                X=X.to(DEVICE),
                y=y.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_loss = self.batch_step(**kwargs)

            # backward pass
            self.backprop(batch_loss)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def batch_step(self, X, y):
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        return loss

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
