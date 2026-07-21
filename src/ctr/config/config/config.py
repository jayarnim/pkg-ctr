from dataclasses import dataclass
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .model import ModelCfg


@dataclass
class Config:
    model: ModelCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    seed: int