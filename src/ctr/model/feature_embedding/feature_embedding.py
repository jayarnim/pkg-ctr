import torch
import torch.nn as nn
from ...featuremap import FeatureMap
from .embedding.builder import build_embedding


class FeatureEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_map: FeatureMap,
    ):
        super().__init__()

        self.feature_map = feature_map

        components = {
            feature.name: build_embedding(
                name=feature.type,
                num_embeddings=feature.vocab_size,
                embedding_dim=embedding_dim,
            )
            for feature in feature_map
        }
        self.embedding = nn.ModuleDict(components)

    def forward(
        self, 
        X: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            self.embedding[feature.name](X[:, feature.column])
            for feature in self.feature_map
        ]