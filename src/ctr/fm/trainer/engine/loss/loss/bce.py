import torch.nn.functional as F


class BinaryCrossEntropy(object):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, y_pred, y):
        kwargs = dict(
            input=y_pred, 
            target=y, 
            reduction='mean',
        )
        return F.binary_cross_entropy_with_logits(**kwargs)