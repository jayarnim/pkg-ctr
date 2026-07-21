from .feature_embedding import FeatureEmbeddingLayer


def build(**kwargs) -> FeatureEmbeddingLayer:
    return FeatureEmbeddingLayer(**kwargs)