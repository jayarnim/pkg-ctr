from .feature_embedding import FeatureEmbeddingLayer


def build_feature_embedding(**kwargs) -> FeatureEmbeddingLayer:
    return FeatureEmbeddingLayer(**kwargs)