"""Unit tests for Phase 1 Agent Contract and Python Orchestrator."""

import pytest
from pydantic import ValidationError

from ai_healthcare.orchestration.orchestrator import (
    AgentNotEnabledError,
    Orchestrator,
)
from ai_healthcare.schemas.agent import CommanderPlan


def test_valid_single_agent_plan_is_accepted() -> None:
    payload = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve relevant evidence about latex allergy.",
                "depends_on": [],
            }
        ],
    }

    plan = Orchestrator().validate_plan(payload)

    assert plan.intent == "medical_information"
    assert plan.tasks[0].agent == "medical_rag"


def test_multi_agent_plan_is_ordered_by_dependencies() -> None:
    payload = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_2",
                "agent": "medication",
                "objective": "Review medication information after symptom extraction.",
                "depends_on": ["task_1"],
            },
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract the symptoms explicitly stated by the user.",
                "depends_on": [],
            },
        ],
    }

    ordered = Orchestrator().get_execution_order(payload)

    assert [task.task_id for task in ordered] == ["task_1", "task_2"]


def test_unknown_agent_is_rejected_by_schema() -> None:
    payload = {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "diagnosis_agent",
                "objective": "Perform an unsupported agent action.",
            }
        ],
    }

    with pytest.raises(ValidationError):
        Orchestrator().validate_plan(payload)


def test_unknown_dependency_is_rejected() -> None:
    payload = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract stated symptoms from the request.",
                "depends_on": ["task_2"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="unknown dependencies"):
        CommanderPlan.model_validate(payload)


def test_dependency_cycle_is_rejected() -> None:
    payload = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract stated symptoms from the request.",
                "depends_on": ["task_2"],
            },
            {
                "task_id": "task_2",
                "agent": "medical_rag",
                "objective": "Retrieve evidence after symptom extraction.",
                "depends_on": ["task_1"],
            },
        ],
    }

    with pytest.raises(ValidationError, match="cycle"):
        CommanderPlan.model_validate(payload)


def test_duplicate_task_ids_are_rejected() -> None:
    payload = {
        "intent": "mixed",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract stated symptoms from the request.",
            },
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve evidence relevant to the request.",
            },
        ],
    }

    with pytest.raises(ValidationError, match="unique"):
        CommanderPlan.model_validate(payload)


def test_ready_plan_requires_at_least_one_task() -> None:
    with pytest.raises(ValidationError, match="at least one task"):
        CommanderPlan.model_validate(
            {"intent": "medical_information", "status": "ready"}
        )


def test_clarification_plan_requires_question_and_no_tasks() -> None:
    with pytest.raises(ValidationError, match="non-empty question"):
        CommanderPlan.model_validate(
            {"intent": "medical_information", "status": "needs_clarification"}
        )

    with pytest.raises(ValidationError, match="must not dispatch"):
        CommanderPlan.model_validate(
            {
                "intent": "medical_information",
                "status": "needs_clarification",
                "clarification_question": "Which medicine do you mean?",
                "tasks": [
                    {
                        "task_id": "task_1",
                        "agent": "medication",
                        "objective": "Identify the medicine mentioned by the user.",
                    }
                ],
            }
        )


def test_unsupported_plan_has_no_tasks() -> None:
    payload = {"intent": "unsupported", "status": "unsupported"}

    plan = Orchestrator().validate_plan(payload)

    assert plan.tasks == ()
    assert Orchestrator().get_execution_order(payload) == ()


def test_disabled_agent_is_rejected_by_orchestrator() -> None:
    payload = {
        "intent": "medication_question",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medication",
                "objective": "Review the medication information provided by the user.",
            }
        ],
    }

    orchestrator = Orchestrator(enabled_agents={"medical_rag", "symptom"})

    with pytest.raises(AgentNotEnabledError, match="not enabled"):
        orchestrator.validate_plan(payload)


def test_orchestrator_cannot_enable_unknown_agent() -> None:
    with pytest.raises(ValueError, match="unknown agents"):
        Orchestrator(enabled_agents={"medical_rag", "free_form_agent"})


def test_extra_fields_are_rejected() -> None:
    payload = {
        "intent": "medical_information",
        "status": "ready",
        "unexpected_tool": "run_shell_command",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve evidence relevant to the question.",
            }
        ],
    }

    with pytest.raises(ValidationError):
        CommanderPlan.model_validate(payload)
