import torch.nn as nn
from .monitor import Monitor
from .predictor import Predictor
from .calculator import Calculator
from .early_stop import EarlyStopping


def monitor_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        model=model,
    )
    predictor = Predictor(model)

    kwargs = dict()
    calculator = Calculator(**kwargs)

    kwargs = dict(
        delta=cfg.delta,
        patience=cfg.patience,
        warmup=cfg.warmup,
    )
    early_stop = EarlyStopping(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
        early_stop=early_stop,
    )
    return Monitor(**kwargs)