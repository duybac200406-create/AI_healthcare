from __future__ import annotations
from ai_healthcare.providers.base import GenerationConfig
import json
from collections.abc import Callable
from typing import Any
import pytest
from pydantic import ValidationError

from ai_healthcare.schemas.agent import AgentTask, CommanderPlan
import httpx
import pytest

from ai_healthcare.agents import (
    Commander,
    CommanderError,
    CommanderInvalidJSONError,
    CommanderPolicyError,
    CommanderSchemaError,
)
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.providers import (
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaProvider,
    OllamaProviderError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from ai_healthcare.schemas.agent import CommanderPlan

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


# --------------------------------------------------------------------------- #
# OllamaProvider
# --------------------------------------------------------------------------- #


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(client=client, **kwargs)


def recording_handler(
    captured: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": '{"x": "1"}'}}
        )

    return handler


def test_provider_sends_expected_request_with_defaults() -> None:
    captured: list[httpx.Request] = []
    provider = make_provider(recording_handler(captured))

    output = provider.generate_json("SYS", "USER", SCHEMA)

    assert output == '{"x": "1"}'
    request = captured[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is False
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 512}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("SYS")
    assert "/no_think" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "USER"}


def test_provider_forwards_custom_configuration() -> None:
    captured: list[httpx.Request] = []
    provider = make_provider(
        recording_handler(captured),
        base_url="http://example.local:9999/",
        model="other:1b",
        temperature=0.2,
        num_ctx=2048,
        num_predict=64,
        no_think=False,
    )

    provider.generate_json("SYS", "USER", SCHEMA)

    body = json.loads(captured[0].content)
    assert str(captured[0].url) == "http://example.local:9999/api/chat"
    assert body["model"] == "other:1b"
    assert body["options"] == {"temperature": 0.2, "num_ctx": 2048, "num_predict": 64}
    assert body["messages"][0]["content"] == "SYS"


def test_provider_connection_error_does_not_leak_prompt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaConnectionError) as exc_info:
        make_provider(handler).generate_json("SYS", "SECRET-USER-TEXT", SCHEMA)
    assert "SECRET-USER-TEXT" not in str(exc_info.value)


def test_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(OllamaTimeoutError):
        make_provider(handler).generate_json("SYS", "USER", SCHEMA)


def test_provider_http_error_carries_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    with pytest.raises(OllamaHTTPError) as exc_info:
        make_provider(handler).generate_json("SYS", "USER", SCHEMA)
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"message": {}}),
        httpx.Response(200, json={"message": {"content": "   "}}),
        httpx.Response(200, json={"message": {"content": 123}}),
    ],
)
def test_provider_rejects_invalid_response_structure(response: httpx.Response) -> None:
    provider = make_provider(lambda request: response)
    with pytest.raises(OllamaResponseError):
        provider.generate_json("SYS", "USER", SCHEMA)


def test_provider_does_not_close_injected_client() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    OllamaProvider(client=client).close()
    assert not client.is_closed


# --------------------------------------------------------------------------- #
# Commander Agent
# --------------------------------------------------------------------------- #


class FakeProvider:
    """Records calls and returns a canned string or raises a canned error."""

    def __init__(self, output: str = "", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.configs = []
    def generate_json(
        self,
        system_prompt,
        user_prompt,
        json_schema,
        *,
        config=None,
    ):  
        self.configs.append(config)
        self.calls.append((system_prompt, user_prompt, json_schema))
        if self.error is not None:
            raise self.error
        return self.output


def ready_plan_json(agent: str = "medical_rag", intent: str = "medical_information") -> str:
    return json.dumps(
        {
            "intent": intent,
            "status": "ready",
            "tasks": [
                {
                    "task_id": "task_1",
                    "agent": agent,
                    "objective": "Retrieve general information about the topic.",
                    "depends_on": [],
                }
            ],
            "clarification_question": None,
        }
    )


def test_commander_builds_schema_from_commander_plan() -> None:
    provider = FakeProvider(ready_plan_json())
    commander = Commander(provider)

    commander.plan("Latex allergy là gì?")

    assert provider.calls[0][2] == CommanderPlan.model_json_schema()
    assert commander.json_schema == CommanderPlan.model_json_schema()


def test_commander_calls_provider_once_with_prompt_schema_and_message() -> None:
    provider = FakeProvider(ready_plan_json())
    commander = Commander(provider)

    commander.plan("  Latex allergy là gì?  ")

    assert len(provider.calls) == 1
    system_prompt, user_prompt, _ = provider.calls[0]
    assert user_prompt == "Latex allergy là gì?"
    assert "Commander Agent" in system_prompt  # loaded from prompts/commander_system.md
    assert "Commander Agent" in system_prompt
    assert "medical_rag" in system_prompt
    assert "symptom" in system_prompt
    assert "medication" in system_prompt
    assert "psychology_support" in system_prompt
    assert '"CommanderPlan"' in system_prompt  # schema appended


def test_commander_returns_validated_plan() -> None:
    commander = Commander(FakeProvider(ready_plan_json()))

    plan = commander.plan("Latex allergy là gì?")

    assert isinstance(plan, CommanderPlan)
    assert plan.status == "ready"
    assert [task.agent for task in plan.tasks] == ["medical_rag"]


def test_commander_rejects_malformed_json() -> None:
    commander = Commander(FakeProvider('{"intent": "mixed", '))

    with pytest.raises(CommanderInvalidJSONError) as exc_info:
        commander.plan("Hỏi gì đó")
    assert exc_info.value.raw_output == '{"intent": "mixed", '


def test_commander_rejects_json_that_is_not_an_object() -> None:
    with pytest.raises(CommanderSchemaError):
        Commander(FakeProvider("[1, 2, 3]")).plan("Hỏi gì đó")


def test_commander_rejects_output_that_violates_pydantic_schema() -> None:
    bad = json.dumps({"intent": "medical_information", "status": "ready", "extra": 1})
    with pytest.raises(CommanderSchemaError):
        Commander(FakeProvider(bad)).plan("Hỏi gì đó")


def test_commander_rejects_inconsistent_plan_from_contract_validators() -> None:
    # status=ready but no tasks: caught by CommanderPlan's own consistency rules.
    bad = json.dumps({"intent": "medical_information", "status": "ready", "tasks": []})
    with pytest.raises(CommanderSchemaError):
        Commander(FakeProvider(bad)).plan("Hỏi gì đó")


def test_commander_rejects_plan_that_orchestrator_does_not_allow() -> None:
    orchestrator = Orchestrator(enabled_agents={"medical_rag"})
    provider = FakeProvider(ready_plan_json("medication", "medication_question"))
    commander = Commander(provider, orchestrator=orchestrator)

    with pytest.raises(CommanderPolicyError) as exc_info:
        commander.plan("Ibuprofen có tác dụng phụ gì?")
    assert "medication" in str(exc_info.value)


def test_commander_preserves_needs_clarification_status() -> None:
    output = json.dumps(
        {
            "intent": "medical_information",
            "status": "needs_clarification",
            "tasks": [],
            "clarification_question": "Bạn muốn hỏi về loại thuốc nào?",
        }
    )

    plan = Commander(FakeProvider(output)).plan("Thuốc đó dùng thế nào?")

    assert plan.status == "needs_clarification"
    assert plan.tasks == ()
    assert plan.clarification_question == "Bạn muốn hỏi về loại thuốc nào?"


def test_commander_preserves_unsupported_status_for_patient_record_question() -> None:
    output = json.dumps(
        {"intent": "unsupported", "status": "unsupported", "tasks": []}
    )

    plan = Commander(FakeProvider(output)).plan(
        "Hồ sơ của tôi đang ghi tình trạng dị ứng như thế nào?"
    )

    assert plan.status == "unsupported"
    assert plan.intent == "unsupported"
    assert plan.tasks == ()  # not re-routed to medical_rag

def test_commander_rejects_agent_outside_whitelist() -> None:
    output = ready_plan_json("patient_record", "medical_information")

    provider = FakeProvider(output)
    commander = Commander(provider)

    with pytest.raises(CommanderSchemaError):
        commander.plan("Giải thích kiến thức y khoa chung.")

    assert len(provider.calls) == 1


def test_commander_never_turns_failures_into_unsupported_or_retries() -> None:
    provider = FakeProvider("not json at all")
    commander = Commander(provider)

    with pytest.raises(CommanderError):
        commander.plan("Hỏi gì đó")
    assert len(provider.calls) == 1  # no repair attempt, no retry, no second call


def test_commander_propagates_provider_errors_unchanged() -> None:
    error = OllamaConnectionError("Cannot reach Ollama")
    commander = Commander(FakeProvider(error=error))

    with pytest.raises(OllamaProviderError) as exc_info:
        commander.plan("Hỏi gì đó")
    assert exc_info.value is error

def test_commander_plan_rejects_duplicate_agents():
    with pytest.raises(ValidationError):
        CommanderPlan(
            intent="mixed",
            status="ready",
            tasks=(
                AgentTask(
                    task_id="task_1",
                    agent="medical_rag",
                    objective="Explain latex allergy.",
                ),
                AgentTask(
                    task_id="task_2",
                    agent="medical_rag",
                    objective="Explain allergy symptoms.",
                ),
            ),
        ),
    
@pytest.mark.parametrize("message", ["", "   \n\t"])
def test_commander_rejects_empty_message_without_calling_provider(message: str) -> None:
    provider = FakeProvider(ready_plan_json())
    with pytest.raises(ValueError):
        Commander(provider).plan(message)
    assert provider.calls == []


def test_commander_rejects_non_string_message() -> None:
    with pytest.raises(TypeError):
        Commander(FakeProvider(ready_plan_json())).plan(123)  # type: ignore[arg-type]


def test_commander_only_plans_and_exposes_no_execution() -> None:
    commander = Commander(FakeProvider(ready_plan_json()))
    plan = commander.plan("Latex allergy là gì?")

    assert not hasattr(commander, "execute")
    assert not hasattr(commander, "run_tasks")
    # The returned plan is immutable data; tasks are proposals, not results.
    with pytest.raises(Exception):
        plan.status = "unsupported"  # type: ignore[misc]

def test_commander_normalizes_intent_from_multiple_agents():
    raw_output = """
    {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Explain general latex allergy knowledge.",
                "depends_on": []
            },
            {
                "task_id": "task_2",
                "agent": "symptom",
                "objective": "Organize rash progression.",
                "depends_on": []
            }
        ],
        "clarification_question": null
    }
    """

    commander = Commander(FakeProvider(raw_output))
    plan = commander.plan(
        "Explain latex allergy and organize rash progression."
    )

    assert plan.intent == "mixed"
    assert plan.status == "ready"
    assert [task.agent for task in plan.tasks] == [
        "medical_rag",
        "symptom",
    ]

def test_commander_normalizes_unsupported_intent():
    raw_output = """
    {
        "intent": "medical_information",
        "status": "unsupported",
        "tasks": [],
        "clarification_question": null
    }
    """

    commander = Commander(FakeProvider(raw_output))
    plan = commander.plan(
        "Look up my individual hospital patient record."
    )

    assert plan.intent == "unsupported"
    assert plan.status == "unsupported"
    assert plan.tasks == ()
    assert plan.clarification_question is None
def test_commander_blocks_individual_patient_record_lookup():
    provider = FakeProvider(ready_plan_json())
    commander = Commander(provider)

    plan = commander.plan(
        "Hồ sơ SYN-001 của tôi đang ghi trạng thái dị ứng latex là gì?"
    )

    assert plan.intent == "unsupported"
    assert plan.status == "unsupported"
    assert plan.tasks == ()
    assert plan.clarification_question is None

    # The request must be blocked before calling the LLM.
    assert len(provider.calls) == 0
def test_commander_allows_general_medical_record_education():
    message = "Những thông tin nào thường được ghi trong hồ sơ bệnh án?"

    assert not Commander._is_patient_record_lookup(message)

def test_commander_detects_patient_specific_record_access():
    messages = [
        "Tra cứu hồ sơ bệnh án của bệnh nhân.",
        "Đọc hồ sơ cá nhân của tôi.",
        "Cho tôi biết thông tin trong hồ sơ SYN-001.",
    ]

    for message in messages:
        assert Commander._is_patient_record_lookup(message)

def test_commander_passes_generation_config_to_provider():
    config = GenerationConfig(
        temperature=0.0,
        num_ctx=8192,
        num_predict=256,
    )

    provider = FakeProvider(ready_plan_json())
    commander = Commander(
        provider,
        generation_config=config,
    )

    commander.plan("Latex allergy là gì?")

    assert provider.configs == [config]