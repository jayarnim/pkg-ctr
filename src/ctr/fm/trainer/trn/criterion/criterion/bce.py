from .base import Criterion
import torch.nn.functional as F


class BinaryCrossEntropy(Criterion):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, pred, true):
        kwargs = dict(
            input=pred, 
            target=true, 
            reduction='mean',
        )
        return F.binary_cross_entropy_with_logits(**kwargs)