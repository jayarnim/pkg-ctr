from ..config.trainer import (
    OptimizerCfg,
    LossCfg,
    TrnCfg,
    ValCfg,
    TrainerCfg,
)


def optimizer(cfg):
    return OptimizerCfg(
        name=cfg["optimizer"]["name"],
        params=cfg["optimizer"].get("params") or dict(),
    )


def loss(cfg):
    return LossCfg(
        name=cfg["loss"]["name"],
        params=cfg["loss"].get("params") or dict(),
    )


def trn(cfg):
    return TrnCfg(
        optimizer=optimizer(cfg),
        loss=loss(cfg),
    )


def val(cfg):
    return ValCfg()


def trainer(cfg):
    return TrainerCfg(
        num_epochs=cfg["trainer"]["num_epochs"],
        patience=cfg["trainer"]["patience"],
        delta=cfg["trainer"]["delta"],
        warmup=cfg["trainer"]["warmup"],
        trn=trn(cfg),
        val=val(cfg),
    )