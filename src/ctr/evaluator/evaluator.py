import pandas as pd
import torch


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(object):
    def __init__(self, predictor, calculator):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator

    def __call__(self, dataloader):
        kwargs = dict(
            dataloader=dataloader,
        )
        prob, true = self.predictor(**kwargs)

        data = dict(
            prob=prob.cpu().numpy(),
            true=true.cpu().numpy(),
        )
        result = pd.DataFrame(data)

        kwargs = dict(
            prob=prob,
            true=true,
        )
        metrics_sheet = self.calculator(**kwargs)

        return result, metrics_sheet