import torch
from .predictor import Predictor
from .calculator import Calculator


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(self, predictor, calculator):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator

    def __call__(self, dataloader, state):
        kwargs = dict(
            dataloader=dataloader,
            state=state,
        )
        prob, true = self.predictor(**kwargs)
        
        kwargs = dict(
            prob=prob,
            true=true,
            state=state,
        )
        self.calculator(**kwargs)