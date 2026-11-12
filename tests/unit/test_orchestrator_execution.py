from typing import Any

import pytest

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.schemas.agent import AgentName, AgentTask


class FakeAgent(BaseAgent):
    """Test agent that simulates success or execution failure."""

    def __init__(
        self,
        name: AgentName,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self.output = output
        self.error = error
        self.calls: list[tuple[AgentTask, dict[str, Any]]] = []

    @property
    def name(self) -> AgentName:
        return self._name

    def execute(
        self,
        task: AgentTask,
        context: dict[str, Any],
    ) -> Any:
        self.calls.append((task, context))

        if self.error is not None:
            raise self.error

        return self.output


def test_registered_agent_executes_successfully() -> None:
    agent = FakeAgent(
        name="medical_rag",
        output={"answer": "Evidence retrieved."},
    )

    registry = AgentRegistry()
    registry.register(agent)

    orchestrator = Orchestrator(registry=registry)

    plan = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve relevant medical evidence.",
            }
        ],
    }

    results = orchestrator.execute_plan(
        plan,
        context={
            "request_id": "test-001",
            "user_message": "What are the symptoms of latex allergy?",
        },
    )

    assert len(results) == 1
    assert results[0].status == "succeeded"
    assert results[0].output == {"answer": "Evidence retrieved."}

    executed_context = agent.calls[0][1]
    assert executed_context["request_id"] == "test-001"
    assert executed_context["dependency_results"] == {}
    assert executed_context["user_message"] == (
        "What are the symptoms of latex allergy?"
    )

def test_dependency_output_is_passed_to_next_agent() -> None:
    symptom_agent = FakeAgent(
        name="symptom",
        output={"symptoms": ["rash", "itching"]},
    )

    medical_agent = FakeAgent(
        name="medical_rag",
        output={"evidence": "Evidence result."},
    )

    registry = AgentRegistry()
    registry.register(symptom_agent)
    registry.register(medical_agent)

    orchestrator = Orchestrator(registry=registry)

    plan = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_2",
                "agent": "medical_rag",
                "objective": "Retrieve evidence based on extracted symptoms.",
                "depends_on": ["task_1"],
            },
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract symptoms explicitly stated by the user.",
            },
        ],
    }

    results = orchestrator.execute_plan(plan)

    assert [result.task_id for result in results] == [
        "task_1",
        "task_2",
    ]

    assert all(result.status == "succeeded" for result in results)

    received_context = medical_agent.calls[0][1]
    dependency_result = received_context["dependency_results"]["task_1"]

    assert dependency_result.output == {
        "symptoms": ["rash", "itching"]
    }
    assert dependency_result.status == "succeeded"


def test_unregistered_agent_returns_not_implemented() -> None:
    orchestrator = Orchestrator(registry=AgentRegistry())

    plan = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve relevant medical evidence.",
            }
        ],
    }

    results = orchestrator.execute_plan(plan)

    assert len(results) == 1
    assert results[0].status == "not_implemented"
    assert "not registered" in results[0].error


def test_failed_dependency_is_skipped_but_independent_task_runs() -> None:
    symptom_agent = FakeAgent(
        name="symptom",
        error=RuntimeError("Sensitive internal error detail"),
    )

    medical_agent = FakeAgent(
        name="medical_rag",
        output={"evidence": "Should not execute."},
    )

    medication_agent = FakeAgent(
        name="medication",
        output={"medication_info": "Independent result."},
    )

    registry = AgentRegistry()
    registry.register(symptom_agent)
    registry.register(medical_agent)
    registry.register(medication_agent)

    orchestrator = Orchestrator(registry=registry)

    plan = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_2",
                "agent": "medical_rag",
                "objective": "Retrieve evidence after symptom extraction.",
                "depends_on": ["task_1"],
            },
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract symptoms explicitly stated by the user.",
            },
            {
                "task_id": "task_3",
                "agent": "medication",
                "objective": "Review the medication information independently.",
            },
        ],
    }

    results = orchestrator.execute_plan(plan)

    result_by_id = {
        result.task_id: result
        for result in results
    }

    assert result_by_id["task_1"].status == "failed"
    assert result_by_id["task_2"].status == "skipped"
    assert result_by_id["task_3"].status == "succeeded"

    assert medical_agent.calls == []
    assert len(symptom_agent.calls) == 1
    assert len(medication_agent.calls) == 1

    # Internal exception details must not leak into the result.
    assert "Sensitive internal error detail" not in result_by_id["task_1"].error


def test_reserved_dependency_context_key_is_rejected() -> None:
    orchestrator = Orchestrator(registry=AgentRegistry())

    plan = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve relevant medical evidence.",
            }
        ],
    }

    with pytest.raises(ValueError, match="reserved context key"):
        orchestrator.execute_plan(
            plan,
            context={"dependency_results": {}},
        )
def test_disallowed_context_is_rejected_before_any_agent_executes() -> None:
    agent = FakeAgent(
        name="medical_rag",
        output={"answer": "Should not execute."},
    )

    registry = AgentRegistry()
    registry.register(agent)

    orchestrator = Orchestrator(registry=registry)

    plan = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve relevant medical evidence.",
            }
        ],
    }

    with pytest.raises(ValueError, match="not permitted"):
        orchestrator.execute_plan(
            plan,
            context={
                "user_message": "I have a skin rash.",
                "patient_record": {
                    "medical_history": "Sensitive information"
                },
            },
        )

    assert agent.calls == []