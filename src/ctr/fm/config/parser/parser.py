from ..config.config import Config
from .model import model
from .pipeline import pipeline
from .trainer import trainer


def parser(cfg):
    return Config(
        model=model(cfg),
        pipeline=pipeline(cfg),
        trainer=trainer(cfg),
        model_cls=cfg["model"]["name"],
        seed=cfg["seed"],
    )