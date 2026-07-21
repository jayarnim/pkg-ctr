from .predictor import Predictor
from .calculator import Calculator
from .engine import Engine


def val_builder(model, cfg):
    kwargs = dict(
        predictor=Predictor(model=model),
        calculator=Calculator(),
    )
    return Engine(**kwargs)