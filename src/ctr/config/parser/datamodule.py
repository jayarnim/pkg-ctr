from ..config.datamodule import *


def split(cfg):
    return SplitCfg(
        ratio=cfg["split"]["ratio"],
        shuffle=cfg["split"]["shuffle"],
        seed=cfg["seed"],
    )


def dataloader(cfg):
    return DataloaderCfg(
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
    )


def datamodule(cfg):
    return DataModuleCfg(
        split=split(cfg),
        dataloader=dataloader(cfg),
    )