import numpy as np
from sklearn.preprocessing import StandardScaler


def scaler(split, cols):
    X_trn, y_trn = split["trn"]
    X_val, y_val = split["val"]
    X_tst, y_tst = split["tst"]

    for X in [X_trn, X_val, X_tst]:
        X.loc[:,cols] = np.log1p(X.loc[:,cols])

    scaler = StandardScaler()
    scaler.fit(X_trn[cols])

    for X in [X_trn, X_val, X_tst]:
        X.loc[:,cols] = scaler.transform(X.loc[:,cols])

    return dict(
        trn=(X_trn, y_trn),
        val=(X_val, y_val),
        tst=(X_tst, y_tst),
    )