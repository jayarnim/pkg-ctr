from sklearn.preprocessing import LabelEncoder


def encoder(
    X, 
    cols,
):
    for col in cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    return X