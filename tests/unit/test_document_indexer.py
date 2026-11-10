import hashlib
import unicodedata
from datetime import datetime, timezone

import pytest

from ai_healthcare.ingestion.indexer import (
    DocumentIndexer,
    DocumentIndexingError,
)
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
)
from ai_healthcare.vectorstores.base import VectorStore, VectorDimensionError


class FakeEmbeddingProvider:
    model = "test-embedding"

    def __init__(self, *, dimensions=4, bad_count=False, bad_vector=None, fail_on_call=None):
        self._dimensions = dimensions
        self.bad_count = bad_count
        self.bad_vector = bad_vector
        self.fail_on_call = fail_on_call
        self.calls = []

    @property
    def dimensions(self):
        return self._dimensions

    def embed(self, text):
        return [float(len(text) or 1), 1.0, 2.0, 3.0]

    def embed_many(self, texts):
        self.calls.append(tuple(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated embedding failure")
        vectors = [
            [float(len(text) or 1), 1.0, 2.0, 3.0]
            for text in texts
        ]
        if self.bad_vector is not None:
            vectors[0] = self.bad_vector
        if self.bad_count:
            return vectors[:-1]
        return vectors


class FakeVectorStore(VectorStore):
    embedding_model = "test-embedding"
    vector_size = 4

    def __init__(self):
        self.records = None
        self.search_args = None
        self.search_results = ()

    def ensure_collection(self):
        pass

    def replace_document(self, records):
        self.records = tuple(records)

    def search(self, query_vector, *, limit=5, filters=None, score_threshold=None):
        self.search_args = {
            "query_vector": tuple(query_vector),
            "limit": limit,
            "filters": filters,
            "score_threshold": score_threshold,
        }
        return self.search_results

    def delete_document(self, document_id):
        pass

    def delete_chunks(self, document_id, chunk_ids):
        pass

    def count(self, filters=None):
        return len(self.records or ())

    def close(self):
        pass


def make_document(content="Café latex allergy evidence. " * 8):
    return MedicalDocument(
        document_id="doc_indexer_test",
        title="Synthetic test evidence",
        source_type="synthetic_fixture",
        source_name="Unit test fixture",
        language="vi",
        content=content,
    )


def make_result(*, embedding_model="test-embedding", embedding_dimensions=4):
    return VectorSearchResult(
        point_id="00000000-0000-0000-0000-000000000001",
        score=0.82,
        document_id="doc_indexer_test",
        chunk_id="chunk_indexer_test_0000",
        chunk_index=0,
        content="Café latex",
        title="Synthetic test evidence",
        source_type="synthetic_fixture",
        source_name="Unit test fixture",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        language="vi",
        start_char=0,
        end_char=10,
        raw_content_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
        normalizer_version="normalizer-v1",
        chunker_version="char-chunker-v1",
        pipeline_fingerprint="c" * 64,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )


def test_indexer_normalizes_chunks_embeds_in_batches_and_replaces_document():
    raw_document = make_document("Cafe\u0301 latex evidence. " * 8)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=provider,
        vector_store=store,
        max_chars=24,
        overlap_chars=5,
        batch_size=2,
    )

    ingestion = indexer.index_document(raw_document)

    assert store.records is not None
    assert len(store.records) == ingestion.chunk_count
    assert [len(batch) for batch in provider.calls] == [2] * (len(provider.calls) - 1) + [
        len(provider.calls[-1])
    ]
    assert all(
        record.document.content == unicodedata.normalize("NFC", raw_document.content)
        for record in store.records
    )
    assert all("e\u0301" not in record.chunk.content for record in store.records)
    assert all(record.embedding_model == provider.model for record in store.records)
    assert all(len(record.vector) == store.vector_size for record in store.records)
    assert ingestion.raw_content_sha256 == hashlib.sha256(raw_document.content.encode("utf-8")).hexdigest()
    assert ingestion.normalized_content_sha256 != ingestion.raw_content_sha256
    assert ingestion.max_chars == 24
    assert ingestion.overlap_chars == 5


def test_indexer_batches_all_chunks_before_store_mutation():
    provider = FakeEmbeddingProvider(fail_on_call=2)
    store = FakeVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=provider,
        vector_store=store,
        max_chars=20,
        overlap_chars=2,
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        indexer.index_document(make_document())

    assert store.records is None
    assert len(provider.calls) == 2


def test_indexer_rejects_provider_store_model_mismatch():
    provider = FakeEmbeddingProvider()
    provider.model = "another-model"

    with pytest.raises(ValueError, match="does not match"):
        DocumentIndexer(embedding_provider=provider, vector_store=FakeVectorStore())


def test_indexer_rejects_known_dimension_mismatch():
    provider = FakeEmbeddingProvider(dimensions=3)

    with pytest.raises(VectorDimensionError):
        DocumentIndexer(embedding_provider=provider, vector_store=FakeVectorStore())


@pytest.mark.parametrize(
    "bad_vector",
    [
        [1.0, 2.0],
        [1.0, float("nan"), 2.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, True, 2.0, 3.0],
    ],
)
def test_indexer_rejects_invalid_embedding_before_store_mutation(bad_vector):
    provider = FakeEmbeddingProvider(bad_vector=bad_vector)
    store = FakeVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=provider,
        vector_store=store,
        max_chars=40,
        overlap_chars=4,
    )

    with pytest.raises((DocumentIndexingError, VectorDimensionError)):
        indexer.index_document(make_document())

    assert store.records is None


def test_indexer_rejects_wrong_embedding_count_before_store_mutation():
    provider = FakeEmbeddingProvider(bad_count=True)
    store = FakeVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=provider,
        vector_store=store,
        max_chars=20,
        overlap_chars=2,
        batch_size=2,
    )

    with pytest.raises(DocumentIndexingError, match="count"):
        indexer.index_document(make_document())

    assert store.records is None


def test_indexer_rejects_non_document_input():
    indexer = DocumentIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(TypeError, match="MedicalDocument"):
        indexer.index_document("not a document")


def test_indexer_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        DocumentIndexer(
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(),
            batch_size=0,
        )


def test_vector_store_interface_exposes_model_and_dimension():
    store = FakeVectorStore()
    assert store.embedding_model == "test-embedding"
    assert store.vector_size == 4
