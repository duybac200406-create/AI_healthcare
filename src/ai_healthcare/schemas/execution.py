from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_healthcare.schemas.agent import AgentName


ExecutionStatus = Literal[
    "succeeded",
    "failed",
    "not_implemented",
    "skipped",
]


class AgentExecutionResult(BaseModel):
    """Structured result of executing one agent task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task_[1-9][0-9]*$")
    agent: AgentName
    status: ExecutionStatus

    output: Any | None = None
    error: str | None = None
    skipped_reason: str | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "AgentExecutionResult":
        if self.status == "succeeded":
            if self.error is not None or self.skipped_reason is not None:
                raise ValueError(
                    "A succeeded result cannot contain error or skipped_reason."
                )

        elif self.status == "failed":
            if not self.error or not self.error.strip():
                raise ValueError("A failed result requires a non-empty error.")

            if self.skipped_reason is not None:
                raise ValueError(
                    "A failed result cannot contain skipped_reason."
                )

        elif self.status == "not_implemented":
            if not self.error or not self.error.strip():
                raise ValueError(
                    "A not_implemented result requires a non-empty error."
                )

            if self.output is not None or self.skipped_reason is not None:
                raise ValueError(
                    "A not_implemented result cannot contain output "
                    "or skipped_reason."
                )

        else:  # skipped
            if not self.skipped_reason or not self.skipped_reason.strip():
                raise ValueError(
                    "A skipped result requires a non-empty skipped_reason."
                )

            if self.output is not None or self.error is not None:
                raise ValueError(
                    "A skipped result cannot contain output or error."
                )

        return self
