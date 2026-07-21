from dataclasses import dataclass
from typing import Literal


FeatureType = Literal["numeric", "categorical"]


@dataclass
class FeatureInfo:
    name: str
    type: FeatureType
    column: int
    vocab_size: int