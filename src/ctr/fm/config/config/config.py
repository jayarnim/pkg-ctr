from dataclasses import dataclass
from typing import Literal
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .model import ModelCfg
from ...models import SUPPORTED_MODELS


@dataclass
class Config:
    model: ModelCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    model_cls: SUPPORTED_MODELS
    seed: int