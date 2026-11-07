from collections.abc import Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    model_validator,
)

from ai_healthcare.schemas.execution import AgentExecutionResult


ResponseStatus = Literal[
    "completed",
    "partial",
    "failed",
    "no_results",
]


def _derive_status(
    results: tuple[AgentExecutionResult, ...],
) -> ResponseStatus:
    """Determine the overall status from individual task results."""

    if not results:
        return "no_results"

    succeeded_count = sum(
        result.status == "succeeded"
        for result in results
    )

    if succeeded_count == len(results):
        return "completed"

    if succeeded_count > 0:
        return "partial"

    return "failed"


class AssembledResponse(BaseModel):
    """Structured response containing all agent execution results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ResponseStatus
    results: tuple[AgentExecutionResult, ...]

    @model_validator(mode="after")
    def validate_response(self) -> "AssembledResponse":
        """Ensure the response status matches its execution results."""

        task_ids = [result.task_id for result in self.results]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Execution results must have unique task IDs."
            )

        expected_status = _derive_status(self.results)

        if self.status != expected_status:
            raise ValueError(
                f"Response status must be '{expected_status}' "
                "for the supplied execution results."
            )

        return self

    @computed_field
    @property
    def total_tasks(self) -> int:
        return len(self.results)

    @computed_field
    @property
    def succeeded_count(self) -> int:
        return sum(
            result.status == "succeeded"
            for result in self.results
        )

    @computed_field
    @property
    def failed_count(self) -> int:
        return sum(
            result.status == "failed"
            for result in self.results
        )

    @computed_field
    @property
    def not_implemented_count(self) -> int:
        return sum(
            result.status == "not_implemented"
            for result in self.results
        )

    @computed_field
    @property
    def skipped_count(self) -> int:
        return sum(
            result.status == "skipped"
            for result in self.results
        )


class ResponseAssembler:
    """Combine individual agent results into one structured response."""

    def assemble(
        self,
        results: Sequence[AgentExecutionResult],
    ) -> AssembledResponse:
        """Validate and assemble execution results."""

        if (
            isinstance(results, (str, bytes))
            or not isinstance(results, Sequence)
        ):
            raise TypeError(
                "results must be a sequence of AgentExecutionResult."
            )

        normalized_results = tuple(results)

        if not all(
            isinstance(result, AgentExecutionResult)
            for result in normalized_results
        ):
            raise TypeError(
                "Every item must be an AgentExecutionResult."
            )

        task_ids = [
            result.task_id
            for result in normalized_results
        ]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Execution results must have unique task IDs."
            )

        return AssembledResponse(
            status=_derive_status(normalized_results),
            results=normalized_results,
        )
