"""Contracts for multi-turn conversation handling (Phase 4A).

These models carry no medical knowledge. They describe where a conversation
stands and what the sufficiency gate decided.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_healthcare.orchestration.response_assembler import AssembledResponse
from ai_healthcare.schemas.agent import CommanderPlan
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.schemas.symptom import SymptomExtractionOutput

GateReason = Literal[
    "sufficient",
    "insufficient_attributes",
    "no_symptom_extracted",
    "needs_clarification_from_agent",
    "max_rounds_reached",
    "no_symptom_identified",
]

ConversationPhase = Literal[
    "idle",
    "awaiting_symptom_details",
    "awaiting_routing_clarification",
]

TurnKind = Literal["answer", "clarification", "unsupported", "error"]


class GateDecision(BaseModel):
    """Deterministic decision of the sufficiency gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow: bool
    reason_code: GateReason
    questions: tuple[str, ...] = Field(default=(), max_length=3)
    limited_information: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "GateDecision":
        if self.allow and self.questions:
            raise ValueError("An allowing decision cannot carry questions.")
        if self.limited_information and not self.allow:
            raise ValueError("limited_information requires allow=True.")
        if len(self.questions) != len(set(self.questions)):
            raise ValueError("Questions must not repeat.")
        return self


class ClarificationNeeded(BaseModel):
    """Output returned in place of a Medical RAG answer when the gate blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: GateDecision


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=8000)


class ConversationState(BaseModel):
    """Serializable state passed into and out of ``handle_turn``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: str = Field(min_length=1)
    phase: ConversationPhase = "idle"
    turns: tuple[Turn, ...] = ()
    # User messages of the episode currently being clarified.
    episode_user_texts: tuple[str, ...] = ()
    pending_questions: tuple[str, ...] = ()
    clarification_rounds: int = Field(default=0, ge=0)

    @classmethod
    def new(cls, conversation_id: str) -> "ConversationState":
        return cls(conversation_id=conversation_id)

    @model_validator(mode="after")
    def _consistent(self) -> "ConversationState":
        if self.phase == "idle" and (
            self.episode_user_texts or self.pending_questions
        ):
            raise ValueError("An idle state cannot hold an open episode.")
        if self.phase != "idle" and not self.episode_user_texts:
            raise ValueError("An open episode requires at least one text.")
        return self


class TurnResult(BaseModel):
    """Everything the UI needs to render one turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: TurnKind
    state: ConversationState
    message: str | None = None
    questions: tuple[str, ...] = ()
    proposed_plan: CommanderPlan | None = None
    executed_plan: CommanderPlan | None = None
    assembled: AssembledResponse | None = None
    gate: GateDecision | None = None
    symptom_output: SymptomExtractionOutput | None = None
    medical_output: MedicalRAGOutput | None = None
    coverage_notice: str | None = None
    language_notice: str | None = None
    language_outcome: str | None = None
    latency_ms: int = Field(default=0, ge=0)
