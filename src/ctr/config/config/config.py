from dataclasses import dataclass
from .datamodule import DataModuleCfg
from .trainer import TrainerCfg
from .model import ModelCfg


@dataclass
class Config:
    model: ModelCfg
    datamodule: DataModuleCfg
    trainer: TrainerCfg
    seed: int