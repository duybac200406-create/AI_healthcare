"""Evidence contracts for source-backed medical generation."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_healthcare.schemas.document import DocumentSourceType
from ai_healthcare.schemas.vector import VectorSearchResult


_EVIDENCE_ID_PATTERN = re.compile(r"^E[1-9][0-9]*$")


class EvidenceReference(BaseModel):
    """A retrieved chunk with immutable, application-controlled provenance.

    Evidence IDs are assigned by Python for one generation request. All source
    metadata is copied from the validated VectorSearchResult, never generated
    by the LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    point_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    title: str = Field(min_length=1)
    source_type: DocumentSourceType
    source_name: str = Field(min_length=1)
    source_url: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    language: str = Field(min_length=2)
    section_title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1)
    similarity_score: float
    pipeline_fingerprint: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int = Field(gt=0)

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("similarity_score must be finite.")
        return value

    @model_validator(mode="after")
    def validate_pages(self) -> "EvidenceReference":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must be provided together.")
        if self.page_start is not None and self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start.")
        return self

    @classmethod
    def from_search_result(
        cls,
        evidence_id: str,
        result: VectorSearchResult,
    ) -> "EvidenceReference":
        if not isinstance(result, VectorSearchResult):
            raise TypeError("result must be a VectorSearchResult.")
        return cls(
            evidence_id=evidence_id,
            point_id=result.point_id,
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            chunk_index=result.chunk_index,
            title=result.title,
            source_type=result.source_type,
            source_name=result.source_name,
            source_url=result.source_url,
            doi=result.doi,
            publication_date=result.publication_date,
            language=result.language,
            section_title=result.section_title,
            page_start=result.page_start,
            page_end=result.page_end,
            excerpt=result.content,
            similarity_score=result.score,
            pipeline_fingerprint=result.pipeline_fingerprint,
            embedding_model=result.embedding_model,
            embedding_dimensions=result.embedding_dimensions,
        )

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return only the source data needed for grounding generation.

        Similarity scores and internal hashes are intentionally omitted so the
        model is less likely to misinterpret similarity as confidence.
        """
        return {
            "evidence_id": self.evidence_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "language": self.language,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content": self.excerpt,
        }
