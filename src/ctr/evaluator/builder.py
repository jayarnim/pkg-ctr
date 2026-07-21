from .predictor import Predictor
from .calculator import Calculator
from .evaluator import Evaluator
import torch.nn as nn


def build_evaluator(
    model: nn.Module, 
    threshold: float,
) -> Evaluator:
    kwargs = dict(
        model=model,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        threshold=threshold,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Evaluator(**kwargs)