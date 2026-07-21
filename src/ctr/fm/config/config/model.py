from dataclasses import dataclass
from ...models import SUPPORTED_MODELS


@dataclass
class ModelCfg:
    name: SUPPORTED_MODELS
    params: dict