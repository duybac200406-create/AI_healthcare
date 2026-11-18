"""Document indexer that uses the structure-aware chunker.

``DocumentIndexer`` is left untouched. This subclass replaces only the
chunking step of ``index_document`` (same normalization, embedding validation
and atomic activation) and records ``structured-chunker-v1`` in the
ingestion fingerprint, so collections built with different chunkers can never
be confused.
"""

from __future__ import annotations

from ai_healthcare.ingestion.indexer import DocumentIndexer, DocumentIndexingError
from ai_healthcare.ingestion.normalizer import NORMALIZER_VERSION, normalize_document
from ai_healthcare.ingestion.structured_chunker import (
    STRUCTURED_CHUNKER_VERSION,
    chunk_document_structured,
)
from ai_healthcare.providers.base import EmbeddingProvider
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.ingestion import IngestionRecord
from ai_healthcare.schemas.vector import VectorRecord
from ai_healthcare.vectorstores.base import VectorStore


class StructuredDocumentIndexer(DocumentIndexer):
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        max_chars: int = 1200,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            max_chars=max_chars,
            overlap_chars=0,
            batch_size=batch_size,
        )

    def index_document(self, document: MedicalDocument) -> IngestionRecord:
        if not isinstance(document, MedicalDocument):
            raise TypeError("document must be a MedicalDocument instance.")

        normalized_document = normalize_document(document)
        chunks = chunk_document_structured(
            normalized_document, max_chars=self._max_chars
        )
        if not chunks:
            raise DocumentIndexingError("Document produced no indexable chunks.")

        ingestion = IngestionRecord.from_documents(
            raw_document=document,
            normalized_document=normalized_document,
            chunks=chunks,
            normalizer_version=NORMALIZER_VERSION,
            chunker_version=STRUCTURED_CHUNKER_VERSION,
            max_chars=self._max_chars,
            overlap_chars=0,
        )

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(chunk_texts), self._batch_size):
            batch = chunk_texts[start : start + self._batch_size]
            embeddings.extend(
                self._validate_embedding_batch(
                    self._embedding_provider.embed_many(batch),
                    expected_count=len(batch),
                )
            )

        if len(embeddings) != len(chunks):
            raise DocumentIndexingError(
                "Embedding count does not match the number of document chunks."
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

        self._vector_store.replace_document(records)
        return ingestion
