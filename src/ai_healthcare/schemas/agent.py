"""Contracts shared by the Commander and specialized agents.

This module defines plans only. It does not execute agents or grant access to
patient records, medical evidence, external tools, or model providers.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentName = Literal[
    "medical_rag",
    "symptom",
    "medication",
    "psychology_support",
]

IntentName = Literal[
    "medical_information",
    "symptom_question",
    "medication_question",
    "psychological_support",
    "mixed",
    "unsupported",
]

PlanStatus = Literal["ready", "needs_clarification", "unsupported"]

SUPPORTED_AGENTS: frozenset[str] = frozenset(
    {
        "medical_rag",
        "symptom",
        "medication",
        "psychology_support",
    }
)


class AgentTask(BaseModel):
    """A bounded task proposal for exactly one whitelisted agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task_[1-9][0-9]*$")
    agent: AgentName
    objective: str = Field(min_length=5, max_length=500)
    depends_on: tuple[str, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "AgentTask":
        if self.task_id in self.depends_on:
            raise ValueError("A task cannot depend on itself.")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("A task cannot list the same dependency more than once.")
        return self


class CommanderPlan(BaseModel):
    """Validated routing plan returned by the Commander.

    The plan contains task descriptions and dependencies, not patient data,
    retrieved evidence, arbitrary tool calls, or executable code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: IntentName
    status: PlanStatus
    tasks: tuple[AgentTask, ...] = Field(default=(), max_length=4)
    clarification_question: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> "CommanderPlan":
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique within a plan.")

        task_id_set = set(task_ids)
        for task in self.tasks:
            unknown_dependencies = set(task.depends_on) - task_id_set
            if unknown_dependencies:
                raise ValueError(
                    f"Task {task.task_id} references unknown dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )

        self._validate_acyclic_dependencies()

        question = self.clarification_question
        if self.status == "ready":
            if not self.tasks:
                raise ValueError("A ready plan must contain at least one task.")
            if question is not None:
                raise ValueError("A ready plan cannot contain a clarification question.")
            if self.intent == "unsupported":
                raise ValueError("An unsupported intent cannot have ready status.")

        elif self.status == "needs_clarification":
            if self.tasks:
                raise ValueError("A clarification plan must not dispatch agent tasks.")
            if question is None or not question.strip():
                raise ValueError(
                    "A clarification plan must include a non-empty question."
                )
            if self.intent == "unsupported":
                raise ValueError(
                    "An unsupported intent must use unsupported status."
                )

        else:  # unsupported
            if self.intent != "unsupported":
                raise ValueError(
                    "Unsupported status requires unsupported intent."
                )
            if self.tasks:
                raise ValueError("An unsupported plan must not contain agent tasks.")
            if question is not None:
                raise ValueError(
                    "An unsupported plan cannot contain a clarification question."
                )

        return self

    def _validate_acyclic_dependencies(self) -> None:
        """Reject dependency cycles before the orchestrator sees the plan."""
        dependencies = {
            task.task_id: set(task.depends_on) for task in self.tasks
        }
        completed: set[str] = set()
        remaining = set(dependencies)

        while remaining:
            ready = {
                task_id
                for task_id in remaining
                if dependencies[task_id].issubset(completed)
            }
            if not ready:
                raise ValueError("Task dependencies contain a cycle.")
            completed.update(ready)
            remaining.difference_update(ready)
