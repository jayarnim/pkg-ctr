from tqdm import tqdm
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(
        self, 
        model: nn.Module,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.current_epoch = 0

    @torch.no_grad()
    def __call__(
        self, 
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        self.current_epoch += 1

        self.model.eval()

        prob_list = []
        true_list = []

        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {self.current_epoch} VAL"
        )
        for X, y in tqdm(**kwargs):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = self.model(X)
            y_prob = torch.sigmoid(y_pred)
            prob_list.extend(y_prob.cpu().tolist())
            true_list.extend(y.cpu().tolist())

        return dict(
            prob=torch.tensor(data=prob_list, dtype=torch.float32).squeeze(-1),
            true=torch.tensor(data=true_list, dtype=torch.int64).squeeze(-1),
        )