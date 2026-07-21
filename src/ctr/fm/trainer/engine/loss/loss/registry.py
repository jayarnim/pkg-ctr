from .bce import BinaryCrossEntropy


LOSS_FN_REGISTRY = {
    "bce": BinaryCrossEntropy,
}