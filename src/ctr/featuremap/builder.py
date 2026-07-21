from .feature_info import FeatureInfo
from .feature_map import FeatureMap
import pandas as pd


def build_feature_map(
    X: pd.DataFrame,
    cat_cols: list=None,
    num_cols: list=None, 
) -> FeatureMap:
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