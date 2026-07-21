from typing import Literal

MODELS = Literal[
    "fm", 
    "deepfm", 
    "xdeepfm",
]

LOSS_FN = Literal[
    "bce",
]

OPTIMIZERS = Literal[
    "adagrad", 
    "adam", 
    "adamw",
]

