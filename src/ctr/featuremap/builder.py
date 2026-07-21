from .featureInfo import FeatureInfo
from .featureMap import FeatureMap


def feature_map_builder(
    X,
    cat_cols=None,
    num_cols=None, 
):
    categoricals = [
        FeatureInfo(
            name=col,
            type="categorical",
            column=X.columns.get_loc(col),
            vocab_size=X[col].nunique(),
        )
        for col in cat_cols
    ]

    numerics = [
        FeatureInfo(
            name=col,
            type="numeric",
            column=X.columns.get_loc(col),
            vocab_size=1,
        )
        for col in num_cols
    ]

    features = categoricals + numerics

    return FeatureMap(features)