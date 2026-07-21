from .criterion.builder import criterion_builder
from .optimizer.builder import optimizer_builder
from .engine import Engine


def trn_builder(model, cfg):
    kwargs = dict(
        params=model.parameters(),
        cfg=cfg.optimizer,
    )
    optimizer = optimizer_builder(**kwargs)

    kwargs = dict(
        cfg=cfg.loss,
    )
    criterion = criterion_builder(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
    return Engine(**kwargs)