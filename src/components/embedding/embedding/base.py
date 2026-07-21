from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Embedding(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError