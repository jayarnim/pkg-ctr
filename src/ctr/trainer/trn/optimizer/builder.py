from .optimizer.registry import OPTIMIZER_REGISTRY


def optimizer_builder(params, cfg):
    kwargs = dict(
        params=params, 
        **cfg.params,
    )
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**kwargs)