"""Schemas and helpers for tracking document ingestion runs."""

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_healthcare.schemas.document import DocumentChunk, MedicalDocument


PIPELINE_FINGERPRINT_VERSION = "pipeline-fingerprint-v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


def _sha256_text(text: str) -> str:
    """Return the SHA-256 digest of the exact UTF-8 representation."""
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Content cannot be encoded as UTF-8.") from None

    return hashlib.sha256(encoded).hexdigest()


def _calculate_fingerprint(
    *,
    document_id: str,
    raw_content_sha256: str,
    normalized_content_sha256: str,
    normalizer_version: str,
    chunker_version: str,
    max_chars: int,
    overlap_chars: int,
    chunk_count: int,
) -> str:
    """Create a deterministic fingerprint for one document-processing result.

    processed_at is intentionally excluded.
    """
    payload = {
        "fingerprint_version": PIPELINE_FINGERPRINT_VERSION,
        "document_id": document_id,
        "raw_content_sha256": raw_content_sha256,
        "normalized_content_sha256": normalized_content_sha256,
        "normalizer_version": normalizer_version,
        "chunker_version": chunker_version,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "chunk_count": chunk_count,
    }

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class IngestionRecord(BaseModel):
    """Immutable metadata describing how one document was processed."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    document_id: str = Field(min_length=1, max_length=256)

    raw_content_sha256: str
    normalized_content_sha256: str

    normalizer_version: str
    chunker_version: str

    max_chars: int = Field(gt=0, strict=True)
    overlap_chars: int = Field(ge=0, strict=True)
    chunk_count: int = Field(gt=0, strict=True)

    pipeline_fingerprint: str

    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank.")
        return value

    @field_validator("raw_content_sha256", "normalized_content_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("A valid lowercase SHA-256 digest is required.")
        return value

    @field_validator("pipeline_fingerprint")
    @classmethod
    def validate_pipeline_fingerprint(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("A valid lowercase pipeline fingerprint is required.")
        return value

    @field_validator("normalizer_version", "chunker_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not _VERSION_PATTERN.fullmatch(value):
            raise ValueError("Version must be a non-empty version token.")
        return value

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("processed_at must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_pipeline_configuration(self) -> "IngestionRecord":
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")

        expected = _calculate_fingerprint(
            document_id=self.document_id,
            raw_content_sha256=self.raw_content_sha256,
            normalized_content_sha256=self.normalized_content_sha256,
            normalizer_version=self.normalizer_version,
            chunker_version=self.chunker_version,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
            chunk_count=self.chunk_count,
        )

        if self.pipeline_fingerprint != expected:
            raise ValueError("pipeline_fingerprint does not match the ingestion metadata.")

        return self

    @classmethod
    def from_documents(
        cls,
        *,
        raw_document: MedicalDocument,
        normalized_document: MedicalDocument,
        chunks: Sequence[DocumentChunk],
        normalizer_version: str,
        chunker_version: str,
        max_chars: int,
        overlap_chars: int,
        processed_at: datetime | None = None,
    ) -> "IngestionRecord":
        """Build an ingestion record from the raw and processed document.

        normalized_document must be the output of normalize_document().
        chunks must be generated from normalized_document.content.
        """
        if not isinstance(raw_document, MedicalDocument):
            raise TypeError("raw_document must be a MedicalDocument.")

        if not isinstance(normalized_document, MedicalDocument):
            raise TypeError("normalized_document must be a MedicalDocument.")

        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise TypeError("chunks must be a sequence of DocumentChunk objects.")

        if not chunks:
            raise ValueError("At least one chunk is required.")

        if raw_document.document_id != normalized_document.document_id:
            raise ValueError("Raw and normalized documents must have the same document_id.")

        raw_metadata = raw_document.model_dump(exclude={"content"})
        normalized_metadata = normalized_document.model_dump(exclude={"content"})

        if raw_metadata != normalized_metadata:
            raise ValueError("Document metadata changed during normalization.")

        normalized_content = normalized_document.content

        for expected_index, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("Every item in chunks must be a DocumentChunk.")

            if chunk.document_id != normalized_document.document_id:
                raise ValueError("Every chunk must belong to the normalized document.")

            if chunk.chunk_index != expected_index:
                raise ValueError("Chunk indices must be contiguous and ordered from zero.")

            if chunk.start_char is None or chunk.end_char is None:
                raise ValueError("Every chunk must have valid character offsets.")

            if chunk.end_char > len(normalized_content):
                raise ValueError("Chunk offsets exceed the normalized document length.")

            expected_content = normalized_content[chunk.start_char:chunk.end_char]
            if chunk.content != expected_content:
                raise ValueError("Chunk content does not match its normalized document offsets.")

        raw_hash = _sha256_text(raw_document.content)
        normalized_hash = _sha256_text(normalized_content)
        chunk_count = len(chunks)

        fingerprint = _calculate_fingerprint(
            document_id=raw_document.document_id,
            raw_content_sha256=raw_hash,
            normalized_content_sha256=normalized_hash,
            normalizer_version=normalizer_version,
            chunker_version=chunker_version,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            chunk_count=chunk_count,
        )

        record_data: dict[str, Any] = {
            "document_id": raw_document.document_id,
            "raw_content_sha256": raw_hash,
            "normalized_content_sha256": normalized_hash,
            "normalizer_version": normalizer_version,
            "chunker_version": chunker_version,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "chunk_count": chunk_count,
            "pipeline_fingerprint": fingerprint,
        }

        if processed_at is not None:
            record_data["processed_at"] = processed_at

        return cls(**record_data)