
"""Validate Commander plans and execute specialized agent tasks."""

from collections.abc import Collection, Mapping
from typing import Any

from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.schemas.agent import (
    SUPPORTED_AGENTS,
    AgentTask,
    CommanderPlan,
)
from ai_healthcare.schemas.execution import AgentExecutionResult


class AgentNotEnabledError(ValueError):
    """Raised when a plan requests a whitelisted but disabled agent."""


class Orchestrator:
    """Validate plans, resolve dependencies, and dispatch agent tasks."""

    def __init__(
        self,
        enabled_agents: Collection[str] | None = None,
        *,
        registry: AgentRegistry | None = None,
    ) -> None:
        if isinstance(enabled_agents, str):
            raise TypeError(
                "enabled_agents must be a collection of agent names, not a string."
            )

        selected = (
            SUPPORTED_AGENTS
            if enabled_agents is None
            else frozenset(enabled_agents)
        )

        unknown = selected - SUPPORTED_AGENTS
        if unknown:
            raise ValueError(
                f"Cannot enable unknown agents: {sorted(unknown)}"
            )

        if registry is not None and not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an AgentRegistry instance.")

        self._enabled_agents = frozenset(selected)
        self._registry = (
            registry if registry is not None else AgentRegistry()
        )

    @property
    def enabled_agents(self) -> frozenset[str]:
        """Return the immutable set of enabled agents."""
        return self._enabled_agents

    def validate_plan(
        self,
        payload: CommanderPlan | Mapping[str, Any],
    ) -> CommanderPlan:
        """Parse and validate a Commander plan without executing tasks."""

        if isinstance(payload, CommanderPlan):
            plan = payload
        elif isinstance(payload, Mapping):
            plan = CommanderPlan.model_validate(payload)
        else:
            raise TypeError(
                "payload must be a CommanderPlan or a mapping."
            )

        requested_agents = {task.agent for task in plan.tasks}
        disabled_agents = requested_agents - self._enabled_agents

        if disabled_agents:
            raise AgentNotEnabledError(
                "Plan requests agents that are not enabled: "
                f"{sorted(disabled_agents)}"
            )

        return plan

    def get_execution_order(
        self,
        payload: CommanderPlan | Mapping[str, Any],
    ) -> tuple[AgentTask, ...]:
        """Return a stable topological order without executing tasks."""

        plan = self.validate_plan(payload)
        return self._derive_execution_order(plan)

    def _derive_execution_order(
        self,
        plan: CommanderPlan,
    ) -> tuple[AgentTask, ...]:
        """Resolve task dependencies while preserving plan order."""

        pending = {task.task_id: task for task in plan.tasks}
        completed: set[str] = set()
        ordered: list[AgentTask] = []

        while pending:
            ready_ids = [
                task.task_id
                for task in plan.tasks
                if task.task_id in pending
                and set(task.depends_on).issubset(completed)
            ]

            if not ready_ids:
                raise ValueError("Cannot resolve task dependencies.")

            for task_id in ready_ids:
                ordered.append(pending.pop(task_id))
                completed.add(task_id)

        return tuple(ordered)

    def execute_plan(
        self,
        payload: CommanderPlan | Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[AgentExecutionResult, ...]:
        """Execute tasks sequentially and collect their structured results."""

        plan = self.validate_plan(payload)

        if context is None:
            base_context: dict[str, Any] = {}
        elif isinstance(context, Mapping):
            base_context = dict(context)
        else:
            raise TypeError("context must be a mapping or None.")

        if "dependency_results" in base_context:
            raise ValueError(
                "'dependency_results' is a reserved context key."
            )

        ordered_tasks = self._derive_execution_order(plan)

        results: list[AgentExecutionResult] = []
        results_by_task_id: dict[str, AgentExecutionResult] = {}

        for task in ordered_tasks:
            failed_dependencies = [
                dependency_id
                for dependency_id in task.depends_on
                if results_by_task_id[dependency_id].status != "succeeded"
            ]

            if failed_dependencies:
                dependency_details = [
                    (
                        dependency_id,
                        results_by_task_id[dependency_id].status,
                    )
                    for dependency_id in failed_dependencies
                ]

                result = AgentExecutionResult(
                    task_id=task.task_id,
                    agent=task.agent,
                    status="skipped",
                    skipped_reason=(
                        "Dependency task(s) did not succeed: "
                        f"{dependency_details}"
                    ),
                )

            else:
                try:
                    agent = self._registry.get(task.agent)

                except KeyError:
                    result = AgentExecutionResult(
                        task_id=task.task_id,
                        agent=task.agent,
                        status="not_implemented",
                        error=(
                            f"Agent '{task.agent}' is not registered."
                        ),
                    )

                else:
                    task_context = dict(base_context)
                    task_context["dependency_results"] = {
                        dependency_id: results_by_task_id[dependency_id]
                        for dependency_id in task.depends_on
                    }

                    try:
                        output = agent.execute(
                            task=task,
                            context=task_context,
                        )

                    except Exception as exc:
                        # Keep internal exception details out of the
                        # result returned to downstream components.
                        result = AgentExecutionResult(
                            task_id=task.task_id,
                            agent=task.agent,
                            status="failed",
                            error=(
                                f"{type(exc).__name__}: "
                                "Agent execution failed."
                            ),
                        )

                    else:
                        result = AgentExecutionResult(
                            task_id=task.task_id,
                            agent=task.agent,
                            status="succeeded",
                            output=output,
                        )

            results.append(result)
            results_by_task_id[task.task_id] = result

        return tuple(results)
