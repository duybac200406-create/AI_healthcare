"""Validate Commander plans and derive a safe execution order.

Phase 1 deliberately does not call an LLM or execute specialized agents.
Actual agent dispatch, permissions, tools, and patient-data access are later
implementation steps.
"""

from collections.abc import Collection, Mapping
from typing import Any

from ai_healthcare.schemas.agent import (
    SUPPORTED_AGENTS,
    AgentTask,
    CommanderPlan,
)


class AgentNotEnabledError(ValueError):
    """Raised when a plan requests a whitelisted but disabled agent."""


class Orchestrator:
    """Validate plans against the allowlist and calculate task order."""

    def __init__(self, enabled_agents: Collection[str] | None = None) -> None:
        if isinstance(enabled_agents, str):
            raise TypeError("enabled_agents must be a collection of agent names, not a string.")

        selected = SUPPORTED_AGENTS if enabled_agents is None else frozenset(enabled_agents)
        unknown = selected - SUPPORTED_AGENTS
        if unknown:
            raise ValueError(f"Cannot enable unknown agents: {sorted(unknown)}")

        self._enabled_agents = frozenset(selected)

    @property
    def enabled_agents(self) -> frozenset[str]:
        """Return the immutable set of agents enabled for this orchestrator."""
        return self._enabled_agents

    def validate_plan(
        self, payload: CommanderPlan | Mapping[str, Any]
    ) -> CommanderPlan:
        """Parse and validate a Commander plan without executing its tasks."""
        if isinstance(payload, CommanderPlan):
            plan = payload
        elif isinstance(payload, Mapping):
            plan = CommanderPlan.model_validate(payload)
        else:
            raise TypeError("payload must be a CommanderPlan or a mapping.")

        requested_agents = {task.agent for task in plan.tasks}
        disabled_agents = requested_agents - self._enabled_agents
        if disabled_agents:
            raise AgentNotEnabledError(
                "Plan requests agents that are not enabled: "
                f"{sorted(disabled_agents)}"
            )

        return plan

    def get_execution_order(
        self, payload: CommanderPlan | Mapping[str, Any]
    ) -> tuple[AgentTask, ...]:
        """Return a stable topological order; do not run the agents."""
        plan = self.validate_plan(payload)
        pending = {task.task_id: task for task in plan.tasks}
        completed: set[str] = set()
        ordered: list[AgentTask] = []

        # Preserve the original plan order among tasks that are ready together.
        while pending:
            ready_ids = [
                task.task_id
                for task in plan.tasks
                if task.task_id in pending
                and set(task.depends_on).issubset(completed)
            ]
            if not ready_ids:
                # CommanderPlan validation should make this unreachable. Keep a
                # defensive failure here in case validation rules later change.
                raise ValueError("Cannot resolve task dependencies.")

            for task_id in ready_ids:
                ordered.append(pending.pop(task_id))
                completed.add(task_id)

        return tuple(ordered)
