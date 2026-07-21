import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: dict[str, float], 
    shuffle: bool, 
    seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    X_opt, X_tst, y_opt, y_tst = train_test_split(
        *(X, y),
        test_size=ratio["tst"],
        shuffle=shuffle,
        stratify=y,
        random_state=seed,
    )

    X_trn, X_val, y_trn, y_val = train_test_split(
        *(X_opt, y_opt),
        test_size=ratio["val"]/(ratio["trn"]+ratio["val"]),
        stratify=y_opt,
        shuffle=shuffle,
        random_state=seed,
    )

    return dict(
        trn=(X_trn, y_trn),
        val=(X_val, y_val),
        tst=(X_tst, y_tst),
    )
