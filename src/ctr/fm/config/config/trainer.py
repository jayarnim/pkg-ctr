from dataclasses import dataclass
from ...const import OPTIMIZERS, LOSS_FN


@dataclass
class OptimizerCfg:
    name: OPTIMIZERS
    params: dict


@dataclass
class LossCfg:
    name: LOSS_FN
    params: dict


@dataclass
class TrnCfg:
    optimizer: OptimizerCfg
    loss: LossCfg


@dataclass
class ValCfg:
    pass


@dataclass
class TrainerCfg:
    num_epochs: int
    patience: int
    delta: float
    warmup: int
    trn: TrnCfg
    val: ValCfg