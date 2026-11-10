"""Provider- and database-neutral schemas for vector storage and retrieval."""

from __future__ import annotations

from datetime import date
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_healthcare.schemas.document import (
    DocumentSourceType,
    MedicalDomain,
    DocumentChunk,
    MedicalDocument,
)
from ai_healthcare.schemas.ingestion import IngestionRecord


_COLLECTION_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,62}$")
_VERSION_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class VectorStoreConfig(BaseModel):
    """Configuration for one dense-vector collection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_name: str = "medical_knowledge_bge_m3_v1"
    vector_size: int = Field(default=1024, gt=0, strict=True)
    distance: Literal["cosine"] = "cosine"
    embedding_model: str = "bge-m3"

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        if not _COLLECTION_PATTERN.fullmatch(value):
            raise ValueError(
                "collection_name must start with a lowercase letter and "
                "contain only lowercase letters, digits, or underscores."
            )
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if not value.strip() or not _VERSION_TOKEN_PATTERN.fullmatch(value):
            raise ValueError("embedding_model must be a non-empty version token.")
        return value


class VectorRecord(BaseModel):
    """One chunk, its source provenance, ingestion metadata, and embedding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: MedicalDocument
    chunk: DocumentChunk
    ingestion: IngestionRecord
    embedding_model: str = Field(min_length=1, max_length=128)
    vector: tuple[float, ...] = Field(min_length=1)

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model must not be blank.")
        return value

    @field_validator("vector", mode="before")
    @classmethod
    def validate_vector_values(cls, value: object) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("vector must be a sequence of numeric values.")
        if not value:
            raise ValueError("vector must not be empty.")

        values: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("vector must contain only numeric values.")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("vector values must be finite.")
            values.append(number)
        return tuple(values)

    @model_validator(mode="after")
    def validate_provenance(self) -> "VectorRecord":
        if self.document.document_id != self.chunk.document_id:
            raise ValueError("document and chunk must have the same document_id.")
        if self.document.document_id != self.ingestion.document_id:
            raise ValueError("document and ingestion record must have the same document_id.")
        if self.chunk.language != self.document.language:
            raise ValueError("chunk language must match the source document language.")
        if self.chunk.start_char is None or self.chunk.end_char is None:
            raise ValueError("vector records require chunk character offsets.")
        if self.chunk.end_char > len(self.document.content):
            raise ValueError("chunk offsets exceed the document content length.")
        if self.document.content[self.chunk.start_char:self.chunk.end_char] != self.chunk.content:
            raise ValueError("chunk content does not match its document offsets.")

        content_hash = hashlib.sha256(self.document.content.encode("utf-8")).hexdigest()
        if content_hash != self.ingestion.normalized_content_sha256:
            raise ValueError("document content does not match the ingestion record.")
        return self

    @property
    def pipeline_fingerprint(self) -> str:
        return self.ingestion.pipeline_fingerprint


class VectorSearchFilter(BaseModel):
    """Allowlisted metadata filters for dense retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_ids: tuple[str, ...] = ()
    source_types: tuple[DocumentSourceType, ...] = ()
    languages: tuple[str, ...] = ()
    medical_domains: tuple[MedicalDomain, ...] = ()

    @field_validator(
        "document_ids", "source_types", "languages", "medical_domains",
        mode="before",
    )
    @classmethod
    def validate_filter_sequence(cls, value: object) -> object:
        if isinstance(value, (str, bytes)):
            raise TypeError("Filter values must be a sequence, not a string.")
        return value

    @model_validator(mode="after")
    def validate_filter_values(self) -> "VectorSearchFilter":
        for field_name in ("document_ids", "source_types", "languages", "medical_domains"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} cannot contain blank values.")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} cannot contain duplicates.")
        return self


class VectorSearchResult(BaseModel):
    """Retrieved chunk with source provenance and Qdrant similarity score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point_id: str
    score: float
    document_id: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    title: str
    source_type: DocumentSourceType
    medical_domain: MedicalDomain = "other"
    source_last_updated: date | None = None
    source_name: str
    source_url: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    retrieved_at: str
    language: str
    license: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    raw_content_sha256: str
    normalized_content_sha256: str
    normalizer_version: str
    chunker_version: str
    pipeline_fingerprint: str
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite.")
        return value

    @model_validator(mode="after")
    def validate_offsets(self) -> "VectorSearchResult":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char.")
        return self
