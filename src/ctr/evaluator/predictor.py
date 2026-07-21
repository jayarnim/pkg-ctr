from tqdm import tqdm
import torch
from ..dataloader import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(self, model):
        super().__init__()
        self.model = model.to(DEVICE)

    @torch.no_grad()
    def __call__(
        self, 
        dataloader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # evalidation
        self.model.eval()

        # reset prob & true
        prob_list = []
        true_list = []

        # iterable obj
        kwargs = dict(
            iterable=dataloader, 
            desc=f"TST"
        )

        # start batch loop
        for X, y in tqdm(**kwargs):
            X = X.to(DEVICE)
            y_pred = self.model(X)
            y_prob = torch.sigmoid(y_pred)

            prob_list.extend(y_prob.cpu().tolist())
            true_list.extend(y.tolist())

        # list -> tensor
        kwargs = dict(
            data=prob_list, 
            dtype=torch.float32,
        )
        prob_tensor = torch.tensor(**kwargs).squeeze(-1)

        kwargs = dict(
            data=true_list, 
            dtype=torch.float32,
        )
        true_tensor = torch.tensor(**kwargs).squeeze(-1)

        return prob_tensor, true_tensor