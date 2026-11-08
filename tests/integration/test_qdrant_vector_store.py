"""Real-server integration tests. Set RUN_QDRANT_INTEGRATION=1 to enable."""

import os
import uuid

import pytest
from qdrant_client import QdrantClient, models

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION, chunk_document
from ai_healthcare.ingestion.normalizer import NORMALIZER_VERSION, normalize_document
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.ingestion import IngestionRecord
from ai_healthcare.schemas.vector import VectorRecord, VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores import (
    CollectionConfigurationError,
    QdrantVectorStore,
    VectorDimensionError,
    VectorStoreError,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RUN_QDRANT_INTEGRATION=1 to run tests against Qdrant Server.",
)


@pytest.fixture
def store_and_client():
    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    client = QdrantClient(url=url, timeout=10.0)
    collection_name = f"test_medical_{uuid.uuid4().hex[:16]}"
    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=3,
        embedding_model="test-embedding",
    )
    store = QdrantVectorStore(client=client, config=config)
    store.ensure_collection()
    try:
        yield store, client, config, url
    finally:
        client.delete_collection(collection_name=collection_name)
        store.close()
        client.close()


def build_records(document_id: str, content: str, language: str = "en"):
    raw = MedicalDocument(
        document_id=document_id,
        title="Synthetic integration test document",
        source_type="research_article",
        source_name="Synthetic integration fixture",
        language=language,
        content=content,
    )
    normalized = normalize_document(raw)
    max_chars = 42
    overlap_chars = 8
    chunks = chunk_document(
        normalized,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    ingestion = IngestionRecord.from_documents(
        raw_document=raw,
        normalized_document=normalized,
        chunks=chunks,
        normalizer_version=NORMALIZER_VERSION,
        chunker_version=CHUNKER_VERSION,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return tuple(
        VectorRecord(
            document=normalized,
            chunk=chunk,
            ingestion=ingestion,
            embedding_model="test-embedding",
            vector=[1.0, 0.0, 0.0],
        )
        for chunk in chunks
    )


def document_filter(document_id: str) -> VectorSearchFilter:
    return VectorSearchFilter(document_ids=(document_id,))


def search_document(store, document_id: str, limit: int = 100):
    return store.search(
        [1.0, 0.0, 0.0],
        limit=limit,
        filters=document_filter(document_id),
    )


def raw_chunk_count(client, collection_name: str, document_id: str) -> int:
    result = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="chunk"),
                ),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        ),
        exact=True,
    )
    return int(result.count)


def test_collection_upsert_search_filter_and_count(store_and_client):
    store, _, _, _ = store_and_client
    records = build_records(
        "doc_qdrant_basic",
        "Medical evidence retrieval requires traceable source metadata. "
        "This synthetic document contains multiple chunks for testing.",
    )
    store.replace_document(records)

    assert store.count() == len(records)
    results = store.search(
        [1.0, 0.0, 0.0],
        limit=5,
        filters=VectorSearchFilter(
            document_ids=("doc_qdrant_basic",),
            source_types=("research_article",),
            languages=("en",),
        ),
    )
    assert results
    assert results[0].document_id == "doc_qdrant_basic"
    assert results[0].embedding_model == "test-embedding"
    assert results[0].embedding_dimensions == 3


def test_replace_document_removes_old_fingerprint_after_upsert(store_and_client):
    store, client, _, _ = store_and_client
    document_id = "doc_qdrant_replace"
    first_records = build_records(
        document_id,
        "First version of this synthetic evidence document. " * 4,
    )
    store.replace_document(first_records)
    first_fingerprint = first_records[0].pipeline_fingerprint

    second_records = build_records(
        document_id,
        "Second version contains updated evidence and revised wording. " * 4,
    )
    assert second_records[0].pipeline_fingerprint != first_fingerprint
    store.replace_document(second_records)

    assert store.count(document_filter(document_id)) == len(second_records)
    assert raw_chunk_count(client, store.config.collection_name, document_id) == len(second_records)
    results = search_document(store, document_id)
    assert {item.pipeline_fingerprint for item in results} == {
        second_records[0].pipeline_fingerprint
    }


def test_replace_is_idempotent(store_and_client):
    store, client, _, _ = store_and_client
    records = build_records(
        "doc_qdrant_idempotent",
        "Repeated upserts with deterministic point identifiers must be idempotent. " * 3,
    )

    store.replace_document(records)
    store.replace_document(records)

    assert store.count(document_filter("doc_qdrant_idempotent")) == len(records)
    assert raw_chunk_count(client, store.config.collection_name, "doc_qdrant_idempotent") == len(records)


def test_cleanup_failure_keeps_old_points_but_search_returns_only_active_version(
    store_and_client, monkeypatch
):
    store, client, _, _ = store_and_client
    document_id = "doc_qdrant_cleanup_recovery"
    first_records = build_records(
        document_id,
        "Old document version remains stored until cleanup succeeds. " * 4,
    )
    second_records = build_records(
        document_id,
        "New document version is active even when stale cleanup fails. " * 4,
    )
    store.replace_document(first_records)

    original_delete = client.delete

    def fail_delete(*args, **kwargs):
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(client, "delete", fail_delete)
    with pytest.raises(VectorStoreError, match="Failed to replace indexed document"):
        store.replace_document(second_records)
    monkeypatch.setattr(client, "delete", original_delete)

    # Both physical versions exist, but only the manifest-selected version is searchable/countable.
    assert raw_chunk_count(client, store.config.collection_name, document_id) == (
        len(first_records) + len(second_records)
    )
    results = search_document(store, document_id)
    assert results
    assert {item.pipeline_fingerprint for item in results} == {
        second_records[0].pipeline_fingerprint
    }
    assert len(results) == len(second_records)
    assert store.count(document_filter(document_id)) == len(second_records)

    # Retrying the same replacement is idempotent and completes stale cleanup.
    store.replace_document(second_records)
    assert raw_chunk_count(client, store.config.collection_name, document_id) == len(second_records)


def test_state_switch_failure_keeps_previous_version_searchable(store_and_client, monkeypatch):
    store, client, _, _ = store_and_client
    document_id = "doc_qdrant_state_recovery"
    first_records = build_records(
        document_id,
        "Previous document version remains active until state pointer switches. " * 4,
    )
    second_records = build_records(
        document_id,
        "Candidate document version is staged before state pointer update. " * 4,
    )
    store.replace_document(first_records)

    original_upsert = client.upsert

    def fail_state_upsert(*args, **kwargs):
        points = kwargs.get("points")
        if points is None and len(args) >= 2:
            points = args[1]
        point_items = points if isinstance(points, (list, tuple)) else [points]
        if any(
            getattr(point, "payload", None)
            and point.payload.get("record_type") == "document_state"
            for point in point_items
        ):
            raise RuntimeError("simulated state-switch failure")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", fail_state_upsert)
    with pytest.raises(VectorStoreError, match="Failed to replace indexed document"):
        store.replace_document(second_records)
    monkeypatch.setattr(client, "upsert", original_upsert)

    # Candidate chunks are staged, but remain invisible until the active pointer switches.
    results = search_document(store, document_id)
    assert results
    assert {item.pipeline_fingerprint for item in results} == {
        first_records[0].pipeline_fingerprint
    }
    assert store.count(document_filter(document_id)) == len(first_records)

    # A retry activates the candidate and cleans up the old version.
    store.replace_document(second_records)
    results = search_document(store, document_id)
    assert {item.pipeline_fingerprint for item in results} == {
        second_records[0].pipeline_fingerprint
    }
    assert raw_chunk_count(client, store.config.collection_name, document_id) == len(second_records)


def test_delete_chunks_and_document(store_and_client):
    store, _, _, _ = store_and_client
    document_id = "doc_qdrant_delete"
    records = build_records(
        document_id,
        "A longer synthetic document is split into several chunks. " * 5,
    )
    assert len(records) >= 2
    store.replace_document(records)

    store.delete_chunks(document_id, [records[0].chunk.chunk_id])
    assert store.count(document_filter(document_id)) == len(records) - 1

    store.delete_document(document_id)
    assert store.count(document_filter(document_id)) == 0
    assert search_document(store, document_id) == ()



def test_delete_document_chunk_cleanup_failure_fails_closed_and_is_retryable(
    store_and_client, monkeypatch
):
    store, client, _, _ = store_and_client
    document_id = "doc_qdrant_delete_recovery"
    records = build_records(
        document_id,
        "A document whose state pointer is removed before stale chunks are deleted. " * 4,
    )
    store.replace_document(records)

    original_delete = client.delete
    calls = {"count": 0}

    def fail_second_delete(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated chunk cleanup failure")
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(client, "delete", fail_second_delete)
    with pytest.raises(VectorStoreError, match="Failed to delete document"):
        store.delete_document(document_id)
    monkeypatch.setattr(client, "delete", original_delete)

    # The manifest is gone, so orphaned chunks must not leak into retrieval.
    assert search_document(store, document_id) == ()
    assert store.count(document_filter(document_id)) == 0
    assert raw_chunk_count(client, store.config.collection_name, document_id) == len(records)

    # Retrying is safe: deleting a missing state point is idempotent, then chunks are removed.
    store.delete_document(document_id)
    assert raw_chunk_count(client, store.config.collection_name, document_id) == 0


def test_store_survives_client_recreation(store_and_client):
    store, client, config, url = store_and_client
    document_id = "doc_qdrant_client_restart"
    records = build_records(
        document_id,
        "Qdrant server state should be available to a newly created client. " * 4,
    )
    store.replace_document(records)
    store.close()

    reopened_store = QdrantVectorStore(url=url, config=config)
    try:
        reopened_store.ensure_collection()
        results = search_document(reopened_store, document_id)
        assert len(results) == len(records)
        assert reopened_store.count(document_filter(document_id)) == len(records)
    finally:
        reopened_store.close()


def test_store_rejects_wrong_embedding_dimension(store_and_client):
    store, _, _, _ = store_and_client
    records = build_records("doc_qdrant_dim", "A dimension test document.")
    wrong_dimension = records[0].model_copy(update={"vector": (1.0, 0.0)})
    with pytest.raises(VectorDimensionError):
        store.replace_document((wrong_dimension,))


def test_existing_collection_configuration_mismatch_is_rejected(store_and_client):
    store, client, _, _ = store_and_client
    mismatched = QdrantVectorStore(
        client=client,
        config=VectorStoreConfig(
            collection_name=store.config.collection_name,
            vector_size=4,
            embedding_model="test-embedding",
        ),
    )
    with pytest.raises(CollectionConfigurationError):
        mismatched.ensure_collection()
    mismatched.close()
