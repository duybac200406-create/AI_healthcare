from datetime import date, datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


DocumentSourceType = Literal[
    "clinical_guideline",
    "systematic_review",
    "meta_analysis",
    "randomized_controlled_trial",
    "observational_study",
    "case_report",
    "drug_label",
    "public_health_guidance",
    "research_article",
    "health_topic_summary",
    "synthetic_fixture",
    "other",
]

MedicalDomain = Literal[
    "cardiovascular",
    "endocrine_metabolic",
    "respiratory",
    "gastrointestinal",
    "renal",
    "neurology",
    "mental_health",
    "allergy_immunology",
    "other",
]


class MedicalDocument(BaseModel):
    """A source document in the medical knowledge corpus.

    ``source_last_updated`` is the update/review date displayed by the source.
    It is not a publication date and must not be copied into ``publication_date``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(
        pattern=r"^doc_[A-Za-z0-9][A-Za-z0-9._-]{0,123}$"
    )

    title: str = Field(min_length=1, max_length=500)

    source_type: DocumentSourceType

    medical_domain: MedicalDomain = "other"

    source_name: str = Field(min_length=1, max_length=200)

    source_url: HttpUrl | None = None

    doi: str | None = Field(default=None, max_length=200)

    publication_date: date | None = None

    source_last_updated: date | None = None

    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    language: str = Field(
        pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$",
        max_length=35,
    )

    license: str | None = Field(default=None, max_length=200)

    content: str = Field(min_length=1)

    @field_validator("title", "source_name")
    @classmethod
    def validate_non_empty_metadata(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not contain only whitespace.")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Document content must not be empty.")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def validate_timezone_aware(
        cls,
        value: datetime,
    ) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_health_topic_provenance(self) -> "MedicalDocument":
        if self.source_type == "health_topic_summary":
            if self.source_url is None:
                raise ValueError(
                    "health_topic_summary requires an official source_url."
                )
            if self.source_last_updated is None:
                raise ValueError(
                    "health_topic_summary requires source_last_updated."
                )
            if self.medical_domain == "other":
                raise ValueError(
                    "health_topic_summary requires an explicit medical_domain."
                )
            if self.source_url.host.lower() not in {
                "medlineplus.gov",
                "www.medlineplus.gov",
            }:
                raise ValueError(
                    "health_topic_summary is restricted to MedlinePlus Health Topic sources."
                )
        return self


class DocumentChunk(BaseModel):
    """A text chunk derived from a source document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(
        pattern=r"^chunk_[A-Za-z0-9][A-Za-z0-9._-]{0,123}$"
    )

    document_id: str = Field(
        pattern=r"^doc_[A-Za-z0-9][A-Za-z0-9._-]{0,123}$"
    )

    chunk_index: int = Field(ge=0)

    content: str = Field(min_length=1)

    section_title: str | None = Field(
        default=None,
        max_length=500,
    )

    language: str = Field(
        pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$",
        max_length=35,
    )

    page_start: int | None = Field(default=None, ge=1)

    page_end: int | None = Field(default=None, ge=1)

    start_char: int | None = Field(default=None, ge=0)

    end_char: int | None = Field(default=None, ge=0)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk content must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_source_locations(self) -> "DocumentChunk":
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(
                "page_start and page_end must be provided together."
            )

        if (
            self.page_start is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must be greater than or equal to page_start."
            )

        if (self.start_char is None) != (self.end_char is None):
            raise ValueError(
                "start_char and end_char must be provided together."
            )

        if (
            self.start_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError(
                "end_char must be greater than start_char."
            )

        return self
