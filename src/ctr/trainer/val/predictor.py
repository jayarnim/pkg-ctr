from tqdm import tqdm
import torch
import torch.nn as nn
from ...dataloader import DataLoader
from ..state import State


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(
        self, 
        model: nn.Module,
    ):
        super().__init__()
        self.model = model.to(DEVICE)

    @torch.no_grad()
    def __call__(
        self, 
        dataloader: DataLoader, 
        state: State,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # evalidation
        self.model.eval()

        # reset prob & true
        prob_list = []
        true_list = []

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {state.current_epoch}/{state.num_epochs} VAL"
        )

        # start batch loop
        for X, y in tqdm(**kwargs):
            # to gpu
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # prdict
            y_pred = self.model(X)
            y_prob = torch.sigmoid(y_pred)

            # accumulate
            prob_list.extend(y_prob.cpu().tolist())
            true_list.extend(y.cpu().tolist())

        # list -> tensor
        kwargs = dict(
            data=prob_list, 
            dtype=torch.float32,
        )
        prob_tensor = torch.tensor(**kwargs).squeeze(-1)

        kwargs = dict(
            data=true_list, 
            dtype=torch.int64,
        )
        true_tensor = torch.tensor(**kwargs).squeeze(-1)

        return prob_tensor, true_tensor