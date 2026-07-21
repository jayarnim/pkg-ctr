from dataclasses import dataclass
from .dataloader import *


@dataclass
class SplitCfg:
    ratio: dict[str, int]
    shuffle: bool
    seed: int


@dataclass
class DataModuleCfg:
    split: SplitCfg
    dataloader: DataloaderCfg