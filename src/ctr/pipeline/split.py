import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def stratified_split_builder(X, y, cfg):
    X_opt, X_tst, y_opt, y_tst = train_test_split(
        *(X, y),
        test_size=cfg.ratio["tst"],
        shuffle=cfg.shuffle,
        stratify=y,
        random_state=cfg.seed,
    )

    X_trn, X_val, y_trn, y_val = train_test_split(
        *(X_opt, y_opt),
        test_size=cfg.ratio["val"]/(cfg.ratio["trn"]+cfg.ratio["val"]),
        stratify=y_opt,
        shuffle=cfg.shuffle,
        random_state=cfg.seed,
    )

    return dict(
        trn=(X_trn, y_trn),
        val=(X_val, y_val),
        tst=(X_tst, y_tst),
    )
