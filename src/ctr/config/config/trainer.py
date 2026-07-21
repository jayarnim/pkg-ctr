from dataclasses import dataclass
from ...core.config.config.criterion import *
from ...core.config.config.optimizer import *


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    criterion: CriterionCfg


@dataclass
class ValCfg:
    pass


@dataclass
class EarlyStoppingCfg:
    patience: int
    delta: float
    warmup: int


@dataclass
class TrainerCfg:
    num_epochs: int
    early_stopping: EarlyStoppingCfg
    trn: TrnCfg
    val: ValCfg