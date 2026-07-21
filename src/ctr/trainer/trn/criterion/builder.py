from .criterion.registry import CRITERION_REGISTRY


def criterion_builder(cfg):
    cls = CRITERION_REGISTRY[cfg.name]
    return cls(**cfg.params)