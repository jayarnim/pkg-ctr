import torch
from .predictor import Predictor
from .calculator import Calculator
from .evaluator import Evaluator


def evaluator_builder(model, threshold):
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