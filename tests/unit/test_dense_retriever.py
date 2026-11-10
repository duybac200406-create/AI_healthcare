from datetime import datetime, timezone

import pytest

from ai_healthcare.retrieval.dense_retriever import (
    DenseRetriever,
    DenseRetrievalError,
)
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorSearchResult
from ai_healthcare.vectorstores.base import VectorDimensionError, VectorStore


class FakeEmbeddingProvider:
    model = "test-embedding"

    def __init__(self, *, dimensions=4, vector=None):
        self._dimensions = dimensions
        self.vector = vector or [1.0, 2.0, 3.0, 4.0]
        self.last_text = None
        self.calls = 0

    @property
    def dimensions(self):
        return self._dimensions

    def embed(self, text):
        self.calls += 1
        self.last_text = text
        return self.vector

    def embed_many(self, texts):
        return [self.vector for _ in texts]


class FakeVectorStore(VectorStore):
    embedding_model = "test-embedding"
    vector_size = 4

    def __init__(self, results=()):
        self.results = tuple(results)
        self.search_args = None

    def ensure_collection(self):
        pass

    def replace_document(self, records):
        pass

    def search(self, query_vector, *, limit=5, filters=None, score_threshold=None):
        self.search_args = {
            "query_vector": tuple(query_vector),
            "limit": limit,
            "filters": filters,
            "score_threshold": score_threshold,
        }
        return self.results

    def delete_document(self, document_id):
        pass

    def delete_chunks(self, document_id, chunk_ids):
        pass

    def count(self, filters=None):
        return len(self.results)

    def close(self):
        pass


def make_result(*, embedding_model="test-embedding", embedding_dimensions=4):
    return VectorSearchResult(
        point_id="00000000-0000-0000-0000-000000000001",
        score=0.82,
        document_id="doc_dense_test",
        chunk_id="chunk_dense_test_0000",
        chunk_index=0,
        content="Evidence content",
        title="Synthetic evidence",
        source_type="synthetic_fixture",
        source_name="Unit test fixture",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        language="vi",
        start_char=0,
        end_char=16,
        raw_content_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
        normalizer_version="normalizer-v1",
        chunker_version="char-chunker-v1",
        pipeline_fingerprint="c" * 64,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )


def test_retriever_normalizes_query_and_passes_filters_and_options():
    provider = FakeEmbeddingProvider()
    result = make_result()
    store = FakeVectorStore([result])
    retriever = DenseRetriever(embedding_provider=provider, vector_store=store)
    filters = VectorSearchFilter(
        source_types=("synthetic_fixture",),
        languages=("vi",),
    )

    results = retriever.retrieve(
        "Cafe\u0301 allergy?",
        top_k=3,
        filters=filters,
        score_threshold=0.4,
    )

    assert provider.last_text == "Café allergy?"
    assert provider.calls == 1
    assert results == (result,)
    assert store.search_args["limit"] == 3
    assert store.search_args["filters"] == filters
    assert store.search_args["score_threshold"] == 0.4
    assert store.search_args["query_vector"] == (1.0, 2.0, 3.0, 4.0)


def test_retriever_does_not_apply_an_unvalidated_default_score_threshold():
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    retriever = DenseRetriever(embedding_provider=provider, vector_store=store)

    retriever.retrieve("A medical question")

    assert store.search_args["score_threshold"] is None
    assert store.search_args["limit"] == 5


@pytest.mark.parametrize("query", ["", "   ", "\ufeff\r\n"])
def test_retriever_rejects_blank_query(query):
    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="blank"):
        retriever.retrieve(query)


def test_retriever_rejects_non_string_query():
    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(TypeError, match="string"):
        retriever.retrieve(None)


def test_retriever_rejects_provider_store_model_mismatch():
    provider = FakeEmbeddingProvider()
    provider.model = "another-model"

    with pytest.raises(ValueError, match="does not match"):
        DenseRetriever(embedding_provider=provider, vector_store=FakeVectorStore())


def test_retriever_rejects_known_dimension_mismatch():
    with pytest.raises(VectorDimensionError):
        DenseRetriever(
            embedding_provider=FakeEmbeddingProvider(dimensions=3),
            vector_store=FakeVectorStore(),
        )


@pytest.mark.parametrize(
    "vector, expected_exception",
    [
        ([1.0, 2.0], VectorDimensionError),
        ([1.0, float("inf"), 3.0, 4.0], DenseRetrievalError),
        ([0.0, 0.0, 0.0, 0.0], DenseRetrievalError),
        ([1.0, True, 3.0, 4.0], DenseRetrievalError),
    ],
)
def test_retriever_rejects_invalid_query_embedding(vector, expected_exception):
    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(vector=vector),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(expected_exception):
        retriever.retrieve("A medical question")


def test_retriever_rejects_result_from_another_embedding_model():
    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore([make_result(embedding_model="other-model")]),
    )

    with pytest.raises(DenseRetrievalError, match="different embedding model"):
        retriever.retrieve("A medical question")


def test_retriever_rejects_result_with_wrong_dimension():
    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore([make_result(embedding_dimensions=3)]),
    )

    with pytest.raises(VectorDimensionError):
        retriever.retrieve("A medical question")
