from .predictor import Predictor
from .calculator import Calculator
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import ValCfg


def build(
    model: nn.Module, 
    cfg: ValCfg,
) -> Engine:
    kwargs = dict(
        model=model,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict()
    calculator = Calculator(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Engine(**kwargs)