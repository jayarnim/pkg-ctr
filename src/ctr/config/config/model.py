from dataclasses import dataclass
from ...const import MODELS


@dataclass
class ModelCfg:
    name: MODELS
    params: dict