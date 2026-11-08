import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION, chunk_document
from ai_healthcare.ingestion.normalizer import NORMALIZER_VERSION, normalize_document
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.ingestion import IngestionRecord
from ai_healthcare.schemas.vector import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStoreConfig,
)


def make_vector_record():
    raw = MedicalDocument(
        document_id="doc_vector_schema_test",
        title="Vector schema test",
        source_type="research_article",
        source_name="Synthetic test source",
        language="en",
        content="A test document about medical evidence retrieval.",
    )
    normalized = normalize_document(raw)
    max_chars = 100
    overlap_chars = 10
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
    return VectorRecord(
        document=normalized,
        chunk=chunks[0],
        ingestion=ingestion,
        embedding_model="bge-m3",
        vector=[0.1, 0.2, 0.3],
    )


def test_vector_store_config_defaults_to_bge_m3_1024_cosine():
    config = VectorStoreConfig()
    assert config.collection_name == "medical_knowledge_bge_m3_v1"
    assert config.vector_size == 1024
    assert config.distance == "cosine"
    assert config.embedding_model == "bge-m3"


def test_vector_store_config_rejects_invalid_collection_name():
    with pytest.raises(ValidationError):
        VectorStoreConfig(collection_name="Medical Knowledge")


def test_vector_record_accepts_valid_provenance_and_vector():
    record = make_vector_record()
    assert record.document.document_id == record.chunk.document_id
    assert len(record.vector) == 3
    assert record.pipeline_fingerprint == record.ingestion.pipeline_fingerprint


@pytest.mark.parametrize("bad_vector", [[1.0, float("nan")], [1.0, float("inf")], [1.0, True]])
def test_vector_record_rejects_non_finite_or_boolean_values(bad_vector):
    record = make_vector_record()
    with pytest.raises((ValidationError, TypeError, ValueError)):
        VectorRecord(
            document=record.document,
            chunk=record.chunk,
            ingestion=record.ingestion,
            embedding_model=record.embedding_model,
            vector=bad_vector,
        )


def test_vector_record_rejects_content_that_does_not_match_offsets():
    record = make_vector_record()
    altered_chunk = record.chunk.model_copy(update={"content": "different text"})
    with pytest.raises(ValidationError, match="does not match"):
        VectorRecord(
            document=record.document,
            chunk=altered_chunk,
            ingestion=record.ingestion,
            embedding_model=record.embedding_model,
            vector=record.vector,
        )


def test_vector_filter_rejects_string_instead_of_sequence():
    with pytest.raises((ValidationError, TypeError)):
        VectorSearchFilter(languages="en")


def test_vector_filter_rejects_duplicate_values():
    with pytest.raises(ValidationError, match="duplicates"):
        VectorSearchFilter(languages=("en", "en"))


def test_vector_search_result_rejects_non_finite_score():
    payload = {
        "point_id": "00000000-0000-0000-0000-000000000001",
        "score": float("nan"),
        "document_id": "doc_x",
        "chunk_id": "chunk_x",
        "chunk_index": 0,
        "content": "content",
        "title": "title",
        "source_type": "research_article",
        "source_name": "source",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "language": "en",
        "start_char": 0,
        "end_char": 7,
        "raw_content_sha256": hashlib.sha256(b"raw").hexdigest(),
        "normalized_content_sha256": hashlib.sha256(b"content").hexdigest(),
        "normalizer_version": "normalizer-v1",
        "chunker_version": "char-chunker-v1",
        "pipeline_fingerprint": "a" * 64,
        "embedding_model": "bge-m3",
        "embedding_dimensions": 3,
    }
    with pytest.raises(ValidationError):
        VectorSearchResult.model_validate(payload)
