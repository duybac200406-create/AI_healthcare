
import json
from typing import Any

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.agents.commander import Commander
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.orchestration.response_assembler import ResponseAssembler
from ai_healthcare.schemas.agent import AgentName, AgentTask


class FakeProvider:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls = 0

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> str:
        self.calls += 1
        return json.dumps(self.plan, ensure_ascii=False)


class FakeAgent(BaseAgent):
    """Simulate a specialized agent."""

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


def test_complete_multi_agent_workflow() -> None:
    """Commander creates a plan and multiple agents execute successfully."""

    symptom_agent = FakeAgent(
        name="symptom",
        output={
            "symptoms": ["rash", "itching"],
        },
    )

    medical_agent = FakeAgent(
        name="medical_rag",
        output={
            "evidence": ["E1", "E2"],
            "summary": "Retrieved relevant evidence.",
        },
    )

    registry = AgentRegistry()
    registry.register(symptom_agent)
    registry.register(medical_agent)

    orchestrator = Orchestrator(registry=registry)

    provider = FakeProvider(
        plan={
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
    )

    commander = Commander(
        provider,
        orchestrator=orchestrator,
        system_prompt="You are a test Commander.",
    )

    # 1. Commander receives a user request and creates a plan.
    plan = commander.plan(
        "Please help me understand these symptoms and relevant evidence."
    )

    # 2. Orchestrator dispatches the registered agents.
    execution_results = orchestrator.execute_plan(
        plan,
        context={"request_id": "integration-success"},
    )

    # 3. ResponseAssembler combines their results.
    response = ResponseAssembler().assemble(execution_results)

    assert provider.calls == 1

    assert [result.task_id for result in execution_results] == [
        "task_1",
        "task_2",
    ]

    assert all(
        result.status == "succeeded"
        for result in execution_results
    )

    assert response.status == "completed"
    assert response.total_tasks == 2
    assert response.succeeded_count == 2

    # Verify that Medical RAG received the Symptom Agent's output.
    medical_context = medical_agent.calls[0][1]

    dependency_result = medical_context[
        "dependency_results"
    ]["task_1"]

    assert dependency_result.output == {
        "symptoms": ["rash", "itching"],
    }


def test_workflow_handles_agent_failure_and_dependency_skip() -> None:
    """A failed agent does not stop independent tasks from executing."""

    symptom_agent = FakeAgent(
        name="symptom",
        error=RuntimeError("Internal simulated failure"),
    )

    medical_agent = FakeAgent(
        name="medical_rag",
        output={"evidence": "Should not be executed."},
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

    provider = FakeProvider(
        plan={
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
                    "objective": "Review medication information independently.",
                },
            ],
        }
    )

    commander = Commander(
        provider,
        orchestrator=orchestrator,
        system_prompt="You are a test Commander.",
    )

    # 1. Commander generates the execution plan.
    plan = commander.plan(
        "Please review my symptoms and medication information."
    )

    # 2. Orchestrator executes tasks and handles failures.
    execution_results = orchestrator.execute_plan(plan)

    # 3. Assemble the partial results.
    response = ResponseAssembler().assemble(execution_results)

    result_by_id = {
        result.task_id: result
        for result in execution_results
    }

    assert provider.calls == 1

    assert result_by_id["task_1"].status == "failed"
    assert result_by_id["task_2"].status == "skipped"
    assert result_by_id["task_3"].status == "succeeded"

    assert medical_agent.calls == []
    assert len(symptom_agent.calls) == 1
    assert len(medication_agent.calls) == 1

    assert response.status == "partial"
    assert response.total_tasks == 3
    assert response.succeeded_count == 1
    assert response.failed_count == 1
    assert response.skipped_count == 1

    assert (
        "Internal simulated failure"
        not in response.results[0].error
    )
