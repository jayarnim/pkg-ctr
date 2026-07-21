from .predictor import Predictor
from .calculator import Calculator
from .engine import Engine
import torch.nn as nn
from ...config.config.trainer import ValCfg


def build_val(
    model: nn.Module, 
    cfg: ValCfg,
) -> Engine:
    kwargs = dict(
        predictor=Predictor(model=model),
        calculator=Calculator(),
    )
    return Engine(**kwargs)