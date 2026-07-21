from dataclasses import dataclass
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .model import ModelCfg
from ...const import MODELS


@dataclass
class Config:
    model: ModelCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    model_cls: MODELS
    seed: int