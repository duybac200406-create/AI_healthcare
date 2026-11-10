import os
import uuid

import pytest

from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RUN_QDRANT_INTEGRATION=1 to run against Qdrant Server.",
)


class DeterministicEmbeddingProvider:
    model = "integration-test-embedding"

    @property
    def dimensions(self):
        return 4

    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    def embed_many(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_indexer_and_dense_retriever_work_with_qdrant_server():
    suffix = uuid.uuid4().hex[:12]
    collection_name = f"dense_it_{suffix}"
    document_id = f"doc_dense_it_{suffix}"
    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=4,
        embedding_model="integration-test-embedding",
    )
    provider = DeterministicEmbeddingProvider()
    store = QdrantVectorStore(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        config=config,
    )

    try:
        store.ensure_collection()
        indexer = DocumentIndexer(
            embedding_provider=provider,
            vector_store=store,
            max_chars=48,
            overlap_chars=8,
            batch_size=2,
        )
        document = MedicalDocument(
            document_id=document_id,
            title="Synthetic dense retrieval integration fixture",
            source_type="synthetic_fixture",
            source_name="Integration test fixture",
            language="en",
            content=(
                "Latex allergy evidence can include diagnostic testing. "
                "Avoid exposure when latex allergy is confirmed. "
            ) * 3,
        )

        ingestion = indexer.index_document(document)
        assert ingestion.chunk_count > 0
        assert store.count() == ingestion.chunk_count

        retriever = DenseRetriever(
            embedding_provider=provider,
            vector_store=store,
        )
        results = retriever.retrieve("latex allergy diagnosis", top_k=3)

        assert results
        assert len(results) <= 3
        assert all(result.document_id == document_id for result in results)
        assert all(result.embedding_model == provider.model for result in results)
        assert all(result.embedding_dimensions == 4 for result in results)

    finally:
        try:
            store._client.delete_collection(collection_name)
        finally:
            store.close()
