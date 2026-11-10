"""Dense retrieval that embeds a query and delegates vector search."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ai_healthcare.ingestion.normalizer import normalize_text
from ai_healthcare.providers.base import EmbeddingProvider
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorSearchResult
from ai_healthcare.vectorstores.base import VectorDimensionError, VectorStore


class DenseRetrievalError(RuntimeError):
    """Raised when dense retrieval cannot safely execute or validate results."""


class DenseRetriever:
    """Retrieve source-backed chunks using one compatible dense embedding model.

    Similarity scores measure vector similarity only. They are not calibrated
    confidence values and do not establish medical evidence sufficiency.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement the VectorStore interface.")
        if not isinstance(getattr(embedding_provider, "model", None), str):
            raise TypeError("embedding_provider must expose a model name.")
        if not callable(getattr(embedding_provider, "embed", None)):
            raise TypeError("embedding_provider must implement embed().")
        if embedding_provider.model != vector_store.embedding_model:
            raise ValueError(
                "Embedding provider model does not match the vector store model."
            )
        if not embedding_provider.model.strip():
            raise ValueError("embedding_provider model must not be blank.")

        dimensions = embedding_provider.dimensions
        if dimensions is not None and dimensions != vector_store.vector_size:
            raise VectorDimensionError(
                "Embedding provider dimensions do not match the vector store."
            )

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Embed one query and return the nearest active document chunks."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        normalized_query = normalize_text(query)
        if not normalized_query.strip():
            raise ValueError("query must not be blank.")

        if self._embedding_provider.model != self._vector_store.embedding_model:
            raise DenseRetrievalError(
                "Embedding provider model no longer matches the vector store."
            )

        query_vector = self._embedding_provider.embed(normalized_query)
        vector = self._validate_query_embedding(query_vector)

        provider_dimensions = self._embedding_provider.dimensions
        if (
            provider_dimensions is not None
            and provider_dimensions != self._vector_store.vector_size
        ):
            raise VectorDimensionError(
                "Embedding provider dimensions do not match the vector store."
            )

        results = self._vector_store.search(
            vector,
            limit=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )

        for result in results:
            if result.embedding_model != self._vector_store.embedding_model:
                raise DenseRetrievalError(
                    "Vector store returned a result from a different embedding model."
                )
            if result.embedding_dimensions != self._vector_store.vector_size:
                raise VectorDimensionError(
                    "Vector store returned a result with an unexpected embedding dimension."
                )

        return results

    def _validate_query_embedding(self, values: object) -> list[float]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise DenseRetrievalError("Embedding provider returned an invalid query vector.")
        if len(values) != self._vector_store.vector_size:
            raise VectorDimensionError(
                "Query embedding dimension does not match the vector store."
            )

        vector: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DenseRetrievalError(
                    "Embedding provider returned a non-numeric query vector."
                )
            number = float(value)
            if not math.isfinite(number):
                raise DenseRetrievalError(
                    "Embedding provider returned a non-finite query vector."
                )
            vector.append(number)

        if not any(value != 0.0 for value in vector):
            raise DenseRetrievalError("Embedding provider returned a zero query vector.")
        return vector
