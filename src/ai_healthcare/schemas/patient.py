"""Schemas for synthetic patient records."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AllergyStatus = Literal["RECORDED_AS_CONFIRMED", "NOT_ASSESSED"]


class SyntheticPatientRecord(BaseModel):
    """A deliberately narrow synthetic patient record for MVP v0.1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    allergy_name: str = Field(min_length=1)
    allergy_status: AllergyStatus
    severity: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    previous_reactions: list[str] = Field(default_factory=list)
    notes: str = ""
