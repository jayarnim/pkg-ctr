import torch
import torch.nn as nn
from .base import Embedding
from .registry import register


@register("categorical")
class CategoricalEmbedding(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.01,
        )

    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(X.long())