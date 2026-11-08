import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION, chunk_document
from ai_healthcare.ingestion.normalizer import NORMALIZER_VERSION, normalize_document
from ai_healthcare.ingestion.synthetic_evidence_loader import load_synthetic_evidence
from ai_healthcare.schemas.ingestion import IngestionRecord


def make_documents(tmp_path, content=None):
    if content is None:
        content = (
            "Bác sĩ có thể sử dụng khám thực thể và xét nghiệm da, "
            "xét nghiệm máu để chẩn đoán dị ứng latex. "
            "Người bệnh nên trao đổi với bác sĩ về các triệu chứng "
            "và tiền sử phản ứng trước khi thực hiện xét nghiệm."
        )

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps({"E1": content}, ensure_ascii=False),
        encoding="utf-8",
    )

    raw_document = load_synthetic_evidence(evidence_path)[0]
    normalized_document = normalize_document(raw_document)

    max_chars = 48
    overlap_chars = 8
    chunks = chunk_document(
        normalized_document,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    return (
        raw_document,
        normalized_document,
        chunks,
        max_chars,
        overlap_chars,
    )


def make_record(
    raw_document,
    normalized_document,
    chunks,
    max_chars,
    overlap_chars,
    **overrides,
):
    options = {
        "raw_document": raw_document,
        "normalized_document": normalized_document,
        "chunks": chunks,
        "normalizer_version": NORMALIZER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
    }
    options.update(overrides)
    return IngestionRecord.from_documents(**options)


def test_from_documents_calculates_content_hashes(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(
        tmp_path,
        content="\ufeffCafe\u0301 là nội dung thử nghiệm về dị ứng latex.",
    )

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    expected_raw_hash = hashlib.sha256(
        raw.content.encode("utf-8")
    ).hexdigest()

    expected_normalized_hash = hashlib.sha256(
        normalized.content.encode("utf-8")
    ).hexdigest()

    assert record.raw_content_sha256 == expected_raw_hash
    assert record.normalized_content_sha256 == expected_normalized_hash
    assert record.raw_content_sha256 != record.normalized_content_sha256


def test_from_documents_records_chunk_metadata(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    assert record.document_id == normalized.document_id
    assert record.max_chars == max_chars
    assert record.overlap_chars == overlap_chars
    assert record.chunk_count == len(chunks)
    assert record.normalizer_version == NORMALIZER_VERSION
    assert record.chunker_version == CHUNKER_VERSION


def test_processed_at_does_not_change_pipeline_fingerprint(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_time = first_time + timedelta(days=10)

    first = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
        processed_at=first_time,
    )

    second = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
        processed_at=second_time,
    )

    assert first.processed_at != second.processed_at
    assert first.pipeline_fingerprint == second.pipeline_fingerprint


def test_normalizer_version_changes_pipeline_fingerprint(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    first = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    second = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
        normalizer_version="normalizer-v2",
    )

    assert first.pipeline_fingerprint != second.pipeline_fingerprint


def test_chunker_version_changes_pipeline_fingerprint(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    first = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    second = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
        chunker_version="char-chunker-v2",
    )

    assert first.pipeline_fingerprint != second.pipeline_fingerprint


def test_content_change_changes_pipeline_fingerprint(tmp_path):
    raw1, normalized1, chunks1, max_chars, overlap_chars = make_documents(
        tmp_path,
        content="Nội dung tài liệu thứ nhất về dị ứng latex.",
    )

    first = make_record(
        raw1,
        normalized1,
        chunks1,
        max_chars,
        overlap_chars,
    )

    raw2, normalized2, chunks2, max_chars, overlap_chars = make_documents(
        tmp_path,
        content="Nội dung tài liệu thứ hai về dị ứng latex.",
    )

    second = make_record(
        raw2,
        normalized2,
        chunks2,
        max_chars,
        overlap_chars,
    )

    # Same file name means same synthetic document ID.
    assert first.document_id == second.document_id
    assert first.raw_content_sha256 != second.raw_content_sha256
    assert first.pipeline_fingerprint != second.pipeline_fingerprint


def test_from_documents_rejects_different_document_ids(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    mismatched_document = normalized.model_copy(
        update={"document_id": "different-document"},
    )

    with pytest.raises(ValueError, match="same document_id"):
        make_record(
            raw,
            mismatched_document,
            chunks,
            max_chars,
            overlap_chars,
        )


def test_from_documents_rejects_chunk_from_another_document(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    mismatched_chunk = chunks[0].model_copy(
        update={"document_id": "different-document"},
    )
    bad_chunks = (mismatched_chunk, *chunks[1:])

    with pytest.raises(ValueError, match="belong to the normalized document"):
        make_record(
            raw,
            normalized,
            bad_chunks,
            max_chars,
            overlap_chars,
        )


def test_from_documents_rejects_chunk_content_that_does_not_match_offsets(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    mismatched_chunk = chunks[0].model_copy(
        update={"content": "fabricated content"},
    )
    bad_chunks = (mismatched_chunk, *chunks[1:])

    with pytest.raises(ValueError, match="does not match"):
        make_record(
            raw,
            normalized,
            bad_chunks,
            max_chars,
            overlap_chars,
        )


def test_from_documents_rejects_non_contiguous_chunk_indices(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    mismatched_chunk = chunks[0].model_copy(
        update={"chunk_index": 5},
    )
    bad_chunks = (mismatched_chunk, *chunks[1:])

    with pytest.raises(ValueError, match="contiguous"):
        make_record(
            raw,
            normalized,
            bad_chunks,
            max_chars,
            overlap_chars,
        )


def test_from_documents_rejects_empty_chunks(tmp_path):
    raw, normalized, _, max_chars, overlap_chars = make_documents(tmp_path)

    with pytest.raises(ValueError, match="At least one chunk"):
        make_record(
            raw,
            normalized,
            (),
            max_chars,
            overlap_chars,
        )


def test_from_documents_rejects_invalid_overlap(tmp_path):
    raw, normalized, chunks, max_chars, _ = make_documents(tmp_path)

    with pytest.raises(ValidationError):
        make_record(
            raw,
            normalized,
            chunks,
            max_chars,
            max_chars,
        )


def test_record_rejects_invalid_sha256(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    data = record.model_dump()
    data["raw_content_sha256"] = "not-a-sha256"

    with pytest.raises(ValidationError):
        IngestionRecord.model_validate(data)


def test_record_rejects_mismatched_pipeline_fingerprint(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    data = record.model_dump()
    data["pipeline_fingerprint"] = "0" * 64

    with pytest.raises(ValidationError, match="does not match"):
        IngestionRecord.model_validate(data)


def test_record_rejects_naive_processed_at(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    data = record.model_dump()
    data["processed_at"] = datetime(2026, 1, 1)

    with pytest.raises(ValidationError, match="timezone-aware"):
        IngestionRecord.model_validate(data)


def test_record_is_immutable(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    with pytest.raises(ValidationError):
        record.chunk_count = 99


def test_record_rejects_unexpected_fields(tmp_path):
    raw, normalized, chunks, max_chars, overlap_chars = make_documents(tmp_path)

    record = make_record(
        raw,
        normalized,
        chunks,
        max_chars,
        overlap_chars,
    )

    data = record.model_dump()
    data["raw_medical_text"] = "This field must not be accepted."

    with pytest.raises(ValidationError):
        IngestionRecord.model_validate(data)