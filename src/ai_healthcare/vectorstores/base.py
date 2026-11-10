"""Database-neutral vector-store interface and exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_healthcare.schemas.vector import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)


class VectorStoreError(RuntimeError):
    """Base error raised by vector-store implementations."""


class VectorDimensionError(VectorStoreError):
    """Raised when an embedding dimension does not match the collection."""


class CollectionConfigurationError(VectorStoreError):
    """Raised when an existing collection is incompatible with its config."""


class VectorStore(ABC):
    """Contract consumed by retrieval services and agents."""

    @property
    @abstractmethod
    def embedding_model(self) -> str:
        """Name of the embedding model indexed in this store."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Expected dense-vector dimension for this store."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the collection if absent and validate it if already present."""

    @abstractmethod
    def replace_document(self, records: Sequence[VectorRecord]) -> None:
        """Upsert a complete document version, then remove older versions."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the nearest chunks matching the allowlisted metadata filters."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete every indexed version of one document."""

    @abstractmethod
    def delete_chunks(self, document_id: str, chunk_ids: Sequence[str]) -> None:
        """Delete selected chunk IDs belonging to one document."""

    @abstractmethod
    def count(self, filters: VectorSearchFilter | None = None) -> int:
        """Return the exact number of points matching optional filters."""

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by the implementation."""
