from .trn.builder import trn_builder
from .val.builder import val_builder
from .callbacks.earlystopping import EarlyStopping
from .callbacks.logger import Logger
from .callbacks.checkpointer import Checkpointer
from .trainer import Trainer


def trainer_builder(model, cfg):
    kwargs = dict(
        model=model,
        cfg=cfg.trn,
    )
    trn = trn_builder(**kwargs)

    kwargs = dict(
        model=model,
        cfg=cfg.val,
    )
    val = val_builder(**kwargs)

    callbacks = [
        EarlyStopping(patience=cfg.patience, delta=cfg.delta, warmup=cfg.warmup),
        Logger(),
        Checkpointer(),
    ]

    kwargs = dict(
        model=model,
        trn=trn,
        val=val,
        callbacks=callbacks,
        num_epochs=cfg.num_epochs,
    )
    return Trainer(**kwargs)