from . import embedding
from .embedding.base import Embedding
from .embedding.registry import EMBEDDING_REGITSTRY


def build(name, **kwargs) -> Embedding:
    cls = EMBEDDING_REGITSTRY[name]
    return cls(**kwargs)