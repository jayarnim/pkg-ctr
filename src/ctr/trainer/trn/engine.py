from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from ..state import State
from .criterion import Criterion
from .optimizer import Optimizer
from ...dataloader import DataLoader


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        criterion: Criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)

    def __call__(
        self, 
        dataloader: DataLoader, 
        state: State,
    ) -> None:
        # train
        self.model.train()

        # reset epoch loss
        epoch_score = 0.0

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} TRN"
        )

        # start batch loop
        for X, y in tqdm(**kwargs):
            # to gpu
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # forward pass
            with autocast(DEVICE.type):
                y_pred = self.model(X)
                batch_score = self.criterion(y_pred, y)

            # backward pass
            self.backprop(batch_score)

            # accumulate loss
            epoch_score += batch_score.item()

        state.trn_score = epoch_score / len(dataloader)

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
