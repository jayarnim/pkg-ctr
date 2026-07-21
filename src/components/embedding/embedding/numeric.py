import torch
import torch.nn as nn
from .base import Embedding
from .registry import register


@register("numeric")
class NumericEmbedding(Embedding):
    def __init__(
        self, 
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            data=torch.empty(1, embedding_dim),
            requires_grad=True,
        )

        nn.init.normal_(
            tensor=self.weight,
            mean=0.0,
            std=0.01,
        )

    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        return X.unsqueeze(-1) * self.weight