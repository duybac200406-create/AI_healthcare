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

    Text fields ending in '_text' must be exact excerpts from user_message.
    The name field is a concise normalized symptom label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Concise normalized label identifying the core symptom. "
            "Normalize wording when appropriate, but preserve clinically "
            "meaningful symptom subtypes and defining modifiers. "
            "Do not broaden a specific symptom into a less specific label; "
            "for example, preserve 'ho khan' rather than reducing it to 'ho'. "
            "The value does not need to be an exact substring of user_message."
        )
    )
    reported_text: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Exact, uninterrupted substring copied from user_message. "
            "For present symptoms, select the shortest span that identifies "
            "the symptom, preserving defining modifiers while excluding "
            "unnecessary grammatical carriers and reporting expressions. "
            "For absent symptoms, preserve the negation evidence. "
            "For uncertain symptoms, preserve the uncertainty evidence. "
            "Never paraphrase, translate, normalize, or reconstruct this field."
        )
    )
    polarity: SymptomPolarity

    onset_text: str | None = Field(
        ...,
        max_length=200,
        description=(
            "Exact, uninterrupted substring stating when the symptom "
            "began (a start time or triggering event), e.g. "
            "'Từ sáng nay', 'hôm qua', 'sau khi ăn'. "
            "Use null when not explicitly mentioned."
        ),
    )

    duration_text: str | None = Field(
        ...,
        max_length=200,
        description=(
            "Exact, uninterrupted substring stating how long the symptom "
            "has lasted or continued, e.g. 'ba ngày nay', "
            "'khoảng hai tuần', 'liên tục 5 giờ'. "
            "Use null when not explicitly mentioned."
        ),
    )

    severity_text: str | None = Field(
        ...,
        max_length=200,
        description=(
            "Exact, uninterrupted substring describing the explicitly "
            "reported symptom severity, e.g. 'nhẹ', 'dữ dội'. "
            "Use null when not explicitly mentioned."
        ),
    )

    body_site_text: str | None = Field(
        ...,
        max_length=200,
        description=(
            "Exact, uninterrupted substring identifying the reported "
            "body location, e.g. 'ở tay', 'vùng bụng dưới'. "
            "Use null when not explicitly mentioned."
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
