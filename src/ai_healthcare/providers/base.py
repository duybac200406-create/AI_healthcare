"""Provider-neutral contracts for embedding backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural interface implemented by local or remote embedding providers."""

    model: str

    @property
    def dimensions(self) -> int | None:
        """Validated embedding dimension, if the provider has generated one."""
        ...

    def embed(self, text: str) -> list[float]:
        """Generate one dense embedding."""
        ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
        ...
