"""Structured contracts for MedicalRAGAgent generation and output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_healthcare.schemas.evidence import EvidenceReference


GenerationStatus = Literal[
    "answered",
    "insufficient_evidence",
    "conflicting_evidence",
]

MedicalRAGStatus = Literal[
    "answered",
    "no_evidence",
    "insufficient_evidence",
    "conflicting_evidence",
    "generation_rejected",
]

AbstentionReason = Literal[
    "no_evidence",
    "insufficient_evidence",
    "conflicting_evidence",
    "invalid_citations",
]


class GeneratedClaim(BaseModel):
    """One atomic answer statement and the evidence IDs supporting it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "GeneratedClaim":
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids cannot contain blank values.")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids cannot contain duplicates.")
        return self


class MedicalGeneration(BaseModel):
    """LLM-facing JSON contract. It contains IDs, never source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GenerationStatus
    claims: tuple[GeneratedClaim, ...] = Field(default=(), max_length=20)
    reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_generation(self) -> "MedicalGeneration":
        if self.status == "answered":
            if not self.claims:
                raise ValueError("An answered generation requires at least one claim.")
            if self.reason is not None:
                raise ValueError("An answered generation cannot contain an abstention reason.")
        else:
            if self.claims:
                raise ValueError("An abstaining generation must not contain answer claims.")
            if self.reason is None or not self.reason.strip():
                raise ValueError("An abstaining generation requires a reason.")
        return self


class GroundedClaim(BaseModel):
    """Validated claim returned to the application with evidence IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)


class MedicalRAGOutput(BaseModel):
    """Safe application output with source-backed citations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: MedicalRAGStatus
    claims: tuple[GroundedClaim, ...] = Field(default=(), max_length=20)
    citations: tuple[EvidenceReference, ...] = ()
    retrieved_evidence: tuple[EvidenceReference, ...] = ()
    abstention_reason: AbstentionReason | None = None
    generation_model: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_output(self) -> "MedicalRAGOutput":
        retrieved_ids = [item.evidence_id for item in self.retrieved_evidence]
        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError("retrieved_evidence IDs must be unique.")

        citation_ids = [item.evidence_id for item in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation IDs must be unique.")
        if not set(citation_ids).issubset(set(retrieved_ids)):
            raise ValueError("Every citation must belong to retrieved evidence.")

        if self.status == "answered":
            if not self.claims or not self.citations:
                raise ValueError("An answered output requires claims and citations.")
            if self.abstention_reason is not None:
                raise ValueError("An answered output cannot have an abstention reason.")
            used_ids = {
                evidence_id
                for claim in self.claims
                for evidence_id in claim.evidence_ids
            }
            if not used_ids.issubset(set(citation_ids)):
                raise ValueError("Every claim evidence ID must have a citation.")
        else:
            if self.claims or self.citations:
                raise ValueError("An abstaining output cannot contain claims or citations.")
            expected_reason: AbstentionReason = {
                "no_evidence": "no_evidence",
                "insufficient_evidence": "insufficient_evidence",
                "conflicting_evidence": "conflicting_evidence",
                "generation_rejected": "invalid_citations",
            }[self.status]
            if self.abstention_reason != expected_reason:
                raise ValueError("Abstention reason does not match output status.")
            if self.status == "no_evidence" and self.retrieved_evidence:
                raise ValueError("no_evidence output cannot contain retrieved evidence.")
            if self.status != "no_evidence" and not self.retrieved_evidence:
                raise ValueError("This abstention status requires retrieved evidence.")

        return self

    @property
    def answer_text(self) -> str:
        """Render a conservative plain-text response for simple clients."""
        if self.status == "answered":
            return "\n".join(claim.text for claim in self.claims)
        messages = {
            "no_evidence": "Tôi chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi này.",
            "insufficient_evidence": "Các tài liệu tìm được chưa cung cấp đủ bằng chứng để trả lời đáng tin cậy.",
            "conflicting_evidence": "Các tài liệu tìm được có thông tin chưa thống nhất; tôi không thể đưa ra kết luận chắc chắn.",
            "generation_rejected": "Câu trả lời được tạo ra không vượt qua kiểm tra nguồn dẫn, nên hệ thống không sử dụng câu trả lời đó.",
        }
        return messages[self.status]
