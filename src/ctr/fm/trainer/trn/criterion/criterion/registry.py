from .bce import BinaryCrossEntropy


CRITERION_REGISTRY = {
    "bce": BinaryCrossEntropy,
}