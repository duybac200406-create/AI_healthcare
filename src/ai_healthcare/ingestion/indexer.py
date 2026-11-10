"""Build and persist dense vectors for normalized document chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION, chunk_document
from ai_healthcare.ingestion.normalizer import NORMALIZER_VERSION, normalize_document
from ai_healthcare.providers.base import EmbeddingProvider
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.ingestion import IngestionRecord
from ai_healthcare.schemas.vector import VectorRecord
from ai_healthcare.vectorstores.base import VectorDimensionError, VectorStore


class DocumentIndexingError(RuntimeError):
    """Raised when a document cannot be embedded and indexed safely."""


class DocumentIndexer:
    """Normalize, chunk, embed, and atomically activate one document version.

    All embeddings are generated and validated before VectorStore.replace_document
    is called. A failed embedding batch therefore cannot partially replace the
    currently active document version in the vector store.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        max_chars: int = 1200,
        overlap_chars: int = 150,
        batch_size: int = 32,
    ) -> None:
        self._validate_dependencies(embedding_provider, vector_store)

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")

        # Let the canonical chunker validate max_chars and overlap_chars.
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise TypeError("max_chars must be an integer.")
        if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int):
            raise TypeError("overlap_chars must be an integer.")
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be between zero and max_chars - 1.")

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars
        self._batch_size = batch_size

    @staticmethod
    def _validate_dependencies(
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement the VectorStore interface.")
        if not isinstance(getattr(embedding_provider, "model", None), str):
            raise TypeError("embedding_provider must expose a model name.")
        if not embedding_provider.model.strip():
            raise ValueError("embedding_provider model must not be blank.")
        if not callable(getattr(embedding_provider, "embed_many", None)):
            raise TypeError("embedding_provider must implement embed_many().")
        if embedding_provider.model != vector_store.embedding_model:
            raise ValueError(
                "Embedding provider model does not match the vector store model."
            )

        dimensions = embedding_provider.dimensions
        if dimensions is not None and dimensions != vector_store.vector_size:
            raise VectorDimensionError(
                "Embedding provider dimensions do not match the vector store."
            )

    def index_document(self, document: MedicalDocument) -> IngestionRecord:
        """Index one complete source document and return its ingestion record."""
        if not isinstance(document, MedicalDocument):
            raise TypeError("document must be a MedicalDocument instance.")

        normalized_document = normalize_document(document)
        chunks = chunk_document(
            normalized_document,
            max_chars=self._max_chars,
            overlap_chars=self._overlap_chars,
        )
        if not chunks:
            raise DocumentIndexingError("Document produced no indexable chunks.")

        ingestion = IngestionRecord.from_documents(
            raw_document=document,
            normalized_document=normalized_document,
            chunks=chunks,
            normalizer_version=NORMALIZER_VERSION,
            chunker_version=CHUNKER_VERSION,
            max_chars=self._max_chars,
            overlap_chars=self._overlap_chars,
        )

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(chunk_texts), self._batch_size):
            batch_texts = chunk_texts[start : start + self._batch_size]
            batch_vectors = self._embedding_provider.embed_many(batch_texts)
            embeddings.extend(
                self._validate_embedding_batch(
                    batch_vectors,
                    expected_count=len(batch_texts),
                )
            )

        if len(embeddings) != len(chunks):
            # Defensive check in case a non-conforming provider returns unusual data.
            raise DocumentIndexingError(
                "Embedding count does not match the number of document chunks."
            )

        if self._embedding_provider.model != self._vector_store.embedding_model:
            raise DocumentIndexingError(
                "Embedding provider model changed during document indexing."
            )

        provider_dimensions = self._embedding_provider.dimensions
        if (
            provider_dimensions is not None
            and provider_dimensions != self._vector_store.vector_size
        ):
            raise VectorDimensionError(
                "Embedding provider dimensions changed during document indexing."
            )

        records = tuple(
            VectorRecord(
                document=normalized_document,
                chunk=chunk,
                ingestion=ingestion,
                embedding_model=self._embedding_provider.model,
                vector=tuple(vector),
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        )

        # The store handles activation and stale-version cleanup.
        self._vector_store.replace_document(records)
        return ingestion

    def _validate_embedding_batch(
        self,
        vectors: object,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
            raise DocumentIndexingError(
                "Embedding provider returned an invalid batch."
            )
        if len(vectors) != expected_count:
            raise DocumentIndexingError(
                "Embedding count does not match the input batch size."
            )

        validated: list[list[float]] = []
        for vector in vectors:
            if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                raise DocumentIndexingError(
                    "Embedding provider returned an invalid vector."
                )
            if len(vector) != self._vector_store.vector_size:
                raise VectorDimensionError(
                    "Embedding dimension does not match the vector store."
                )

            values: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise DocumentIndexingError(
                        "Embedding provider returned a non-numeric value."
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise DocumentIndexingError(
                        "Embedding provider returned a non-finite value."
                    )
                values.append(number)

            if not any(value != 0.0 for value in values):
                raise DocumentIndexingError(
                    "Embedding provider returned a zero vector."
                )
            validated.append(values)

        return validated
