import os
from datetime import date

import pytest
from qdrant_client import QdrantClient

from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration tests.",
)


class FakeEmbeddingProvider:
    model = "medical-domain-test-embedding"
    dimensions = 3

    def embed(self, text):
        return [float(len(text) or 1), 1.0, 0.25]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


def _make_document(document_id, title, domain, url, updated, content):
    return MedicalDocument(
        document_id=document_id,
        title=title,
        source_type="health_topic_summary",
        medical_domain=domain,
        source_name="MedlinePlus, National Library of Medicine",
        source_url=url,
        source_last_updated=date.fromisoformat(updated),
        publication_date=None,
        language="en",
        license="Public domain MedlinePlus Health Topic Summary; attribution requested.",
        content=content,
    )


def test_qdrant_filters_and_returns_medical_domain_provenance():
    collection = "medical_domain_filter_test"
    client = QdrantClient(url="http://127.0.0.1:6333")
    if client.collection_exists(collection):
        client.delete_collection(collection)
    store = QdrantVectorStore(
        client=client,
        config=VectorStoreConfig(
            collection_name=collection,
            vector_size=3,
            embedding_model=FakeEmbeddingProvider.model,
        ),
    )

    try:
        store.ensure_collection()
        indexer = DocumentIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=store,
            max_chars=500,
            overlap_chars=20,
        )
        indexer.index_document(
            _make_document(
                "doc_medlineplus_domain_heart",
                "Heart Diseases",
                "cardiovascular",
                "https://medlineplus.gov/heartdiseases.html",
                "2025-03-20",
                "Heart disease affects the heart and blood vessels.",
            )
        )
        indexer.index_document(
            _make_document(
                "doc_medlineplus_domain_kidney",
                "Chronic Kidney Disease",
                "renal",
                "https://medlineplus.gov/chronickidneydisease.html",
                "2025-09-21",
                "Chronic kidney disease affects how the kidneys filter blood.",
            )
        )

        results = store.search(
            [1.0, 0.5, 0.25],
            limit=10,
            filters=VectorSearchFilter(medical_domains=("cardiovascular",)),
        )

        assert results
        assert all(item.medical_domain == "cardiovascular" for item in results)
        assert all(item.document_id == "doc_medlineplus_domain_heart" for item in results)
        assert results[0].source_last_updated == date(2025, 3, 20)
        assert store.count(VectorSearchFilter(medical_domains=("cardiovascular",))) == 1
        assert store.count(VectorSearchFilter(medical_domains=("renal",))) == 1
    finally:
        store.close()
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()
