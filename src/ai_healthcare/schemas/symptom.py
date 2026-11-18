from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SymptomPolarity = Literal["present", "absent", "uncertain"]

SymptomExtractionStatus = Literal[
    "extracted",
    "no_symptoms",
    "needs_clarification",
]


class SymptomMention(BaseModel):
    """One symptom explicitly mentioned or denied by the user.

    Every populated field ending in '_text' must be an exact, uninterrupted
    excerpt from user_message. The name field is a concise normalized label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Concise normalized label identifying the core symptom. "
            "Preserve clinically meaningful symptom subtypes and defining "
            "modifiers. Do not infer a disease or diagnosis."
        ),
    )
    reported_text: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Exact, uninterrupted substring copied from user_message. "
            "For present symptoms, select the shortest span identifying the "
            "symptom. For absent and uncertain symptoms, preserve the "
            "negation or uncertainty evidence. Never paraphrase this field."
        ),
    )
    polarity: SymptomPolarity

    onset_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring stating when the symptom began or a triggering "
            "event. Null when not explicitly mentioned."
        ),
    )
    duration_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring stating how long the symptom has lasted. "
            "Do not infer it from an onset phrase. Null when not stated."
        ),
    )
    severity_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring describing explicitly reported intensity, "
            "such as mild, moderate or severe. Frequency is not severity."
        ),
    )
    body_site_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring identifying the body location. Preserve the "
            "complete anatomical region. Null when not stated."
        ),
    )
    quality_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring describing the quality or character of the "
            "symptom, such as sharp, dull, throbbing, burning or cramping. "
            "Do not infer a quality. Null when not stated."
        ),
    )
    pattern_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Exact substring describing the pattern, recurrence or "
            "frequency, such as continuous, intermittent, in episodes or "
            "comes and goes. This is distinct from elapsed duration and "
            "severity. Null when not stated."
        ),
    )
    aggravating_factors_text: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Exact substring describing what makes the symptom worse, "
            "such as movement, eating or coughing. Null when not stated."
        ),
    )
    relieving_factors_text: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Exact substring describing what makes the symptom better, "
            "such as rest or a stated change in position. Do not infer "
            "effective treatments. Null when not stated."
        ),
    )
    functional_impact_text: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Exact substring describing the reported effect on daily "
            "activities, sleep, eating, walking, work or study. Null when "
            "not stated."
        ),
    )

    @field_validator("name", "reported_text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text fields must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_optional_text(self) -> "SymptomMention":
        optional_fields = (
            self.onset_text,
            self.duration_text,
            self.severity_text,
            self.body_site_text,
            self.quality_text,
            self.pattern_text,
            self.aggravating_factors_text,
            self.relieving_factors_text,
            self.functional_impact_text,
        )

        if any(
            value is not None and not value.strip()
            for value in optional_fields
        ):
            raise ValueError("Optional text fields must not be blank.")

        return self


class SymptomExtractionOutput(BaseModel):
    """Validated application output produced by SymptomAgent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SymptomExtractionStatus

    symptoms: tuple[SymptomMention, ...] = Field(
        default=(),
        max_length=30,
    )

    follow_up_questions: tuple[str, ...] = Field(
        default=(),
        max_length=5,
    )

    @field_validator("follow_up_questions")
    @classmethod
    def validate_follow_up_questions(
        cls,
        questions: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not question.strip() for question in questions):
            raise ValueError("Follow-up questions cannot be blank.")
        return questions

    @model_validator(mode="after")
    def validate_extraction(self) -> "SymptomExtractionOutput":
        if self.status == "extracted" and not self.symptoms:
            raise ValueError(
                "An extracted result requires at least one symptom."
            )

        if self.status == "no_symptoms":
            if self.symptoms:
                raise ValueError(
                    "A no_symptoms result must not contain symptoms."
                )
            if self.follow_up_questions:
                raise ValueError(
                    "A no_symptoms result must not contain follow-up questions."
                )

        if (
            self.status == "needs_clarification"
            and not self.follow_up_questions
        ):
            raise ValueError(
                "A needs_clarification result requires follow-up questions."
            )

        fingerprints = [
            (
                item.name.casefold(),
                item.reported_text.casefold(),
                item.polarity,
            )
            for item in self.symptoms
        ]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("Duplicate symptom mentions are not allowed.")

        return self
