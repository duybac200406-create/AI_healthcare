import json

import pytest

from ai_healthcare.agents.symptom import (
    SymptomAgent,
    SymptomGenerationError,
    SymptomInputError,
)
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.providers.base import GenerationConfig
from ai_healthcare.schemas.agent import AgentTask
from ai_healthcare.schemas.symptom import SymptomExtractionOutput


class FakeProvider:
    model = "qwen3:8b"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []
        self.configs = []

    def generate_json(
        self,
        system_prompt,
        user_prompt,
        json_schema,
        *,
        config=None,
    ):
        self.calls.append(
            (system_prompt, user_prompt, json_schema)
        )
        self.configs.append(config)
        return self.output


def make_output(
    *,
    status="extracted",
    symptoms=None,
    follow_up_questions=None,
):
    payload = {
        "status": status,
        "symptoms": symptoms if symptoms is not None else [],
        "follow_up_questions": (
            follow_up_questions
            if follow_up_questions is not None
            else []
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def make_symptom(
    *,
    name="phát ban",
    reported_text="nổi mẩn đỏ",
    polarity="present",
    onset_text=None,
    duration_text=None,
    severity_text=None,
    body_site_text=None,
):
    return {
        "name": name,
        "reported_text": reported_text,
        "polarity": polarity,
        "onset_text": onset_text,
        "duration_text": duration_text,
        "severity_text": severity_text,
        "body_site_text": body_site_text,
    }


def make_task(agent="symptom"):
    return AgentTask(
        task_id="task_1",
        agent=agent,
        objective="Extract explicitly reported symptoms from the message.",
    )


def make_agent(output=None, **kwargs):
    provider = FakeProvider(
        output if output is not None else make_output(
            symptoms=[make_symptom()]
        )
    )
    agent = SymptomAgent(
        provider=provider,
        **kwargs,
    )
    return agent, provider


def test_agent_name():
    agent, _ = make_agent()
    assert agent.name == "symptom"


def test_extracts_symptoms_with_verbatim_source_spans():
    message = "Tôi bị nổi mẩn đỏ và ngứa ở cánh tay từ sáng nay."

    generated = make_output(
        symptoms=[
            make_symptom(
                name="phát ban",
                reported_text="nổi mẩn đỏ",
                onset_text="từ sáng nay",
                body_site_text="ở cánh tay",
            ),
            make_symptom(
                name="ngứa",
                reported_text="ngứa",
                onset_text="từ sáng nay",
                body_site_text="ở cánh tay",
            ),
        ]
    )

    agent, provider = make_agent(generated)

    output = agent.execute(
        make_task(),
        {"user_message": message},
    )

    assert isinstance(output, SymptomExtractionOutput)
    assert output.status == "extracted"
    assert len(output.symptoms) == 2
    assert output.symptoms[0].name == "phát ban"
    assert output.symptoms[0].reported_text == "nổi mẩn đỏ"
    assert output.symptoms[0].body_site_text == "ở cánh tay"
    assert output.symptoms[1].name == "ngứa"
    assert len(provider.calls) == 1


def test_extracts_negated_and_present_symptoms_separately():
    message = "Tôi không sốt nhưng hơi đau đầu."

    generated = make_output(
        symptoms=[
            make_symptom(
                name="sốt",
                reported_text="không sốt",
                polarity="absent",
            ),
            make_symptom(
                name="đau đầu",
                reported_text="đau đầu",
                polarity="present",
            ),
        ]
    )

    agent, _ = make_agent(generated)

    output = agent.execute(
        make_task(),
        {"user_message": message},
    )

    assert output.symptoms[0].polarity == "absent"
    assert output.symptoms[1].polarity == "present"


def test_no_symptoms_is_a_valid_result():
    agent, _ = make_agent(
        make_output(status="no_symptoms")
    )

    output = agent.execute(
        make_task(),
        {"user_message": "Dị ứng latex là gì?"},
    )

    assert output.status == "no_symptoms"
    assert output.symptoms == ()


def test_needs_clarification_requires_questions():
    agent, _ = make_agent(
        make_output(
            status="needs_clarification",
            follow_up_questions=[
                "Bạn có thể mô tả rõ hơn cảm giác đó không?"
            ],
        )
    )

    output = agent.execute(
        make_task(),
        {"user_message": "Tôi cảm thấy có gì đó không ổn."},
    )

    assert output.status == "needs_clarification"
    assert len(output.follow_up_questions) == 1


def test_rejects_source_text_not_present_in_user_message():
    agent, provider = make_agent(
        make_output(
            symptoms=[
                make_symptom(
                    reported_text="sốt",
                )
            ]
        )
    )

    with pytest.raises(
        SymptomGenerationError,
        match="source span",
    ):
        agent.execute(
            make_task(),
            {"user_message": "Tôi bị đau đầu."},
        )

    assert len(provider.calls) == 1


def test_rejects_optional_attribute_not_present_in_source():
    agent, _ = make_agent(
        make_output(
            symptoms=[
                make_symptom(
                    reported_text="nổi mẩn đỏ",
                    duration_text="3 ngày",
                )
            ]
        )
    )

    with pytest.raises(
        SymptomGenerationError,
        match="duration_text",
    ):
        agent.execute(
            make_task(),
            {"user_message": "Tôi bị nổi mẩn đỏ."},
        )


@pytest.mark.parametrize(
    "invalid_output",
    [
        "not-json",
        '{"status":"invalid","symptoms":[],"follow_up_questions":[]}',
        '{"status":"extracted","symptoms":[],"follow_up_questions":[]}',
        '{"status":"no_symptoms","symptoms":[{"name":"sốt"}]}',
    ],
)
def test_rejects_invalid_model_output(invalid_output):
    agent, _ = make_agent(invalid_output)

    with pytest.raises(SymptomGenerationError):
        agent.execute(
            make_task(),
            {"user_message": "Tôi bị đau đầu."},
        )


def test_rejects_patient_record_context():
    agent, provider = make_agent()

    with pytest.raises(
        SymptomInputError,
        match="does not accept patient-record",
    ):
        agent.execute(
            make_task(),
            {
                "user_message": "Tôi bị đau đầu.",
                "patient_record": {"diagnosis": "test"},
            },
        )

    assert provider.calls == []


def test_requires_user_message():
    agent, provider = make_agent()

    with pytest.raises(
        SymptomInputError,
        match="user_message",
    ):
        agent.execute(make_task(), {})

    assert provider.calls == []


def test_rejects_wrong_user_message_type():
    agent, provider = make_agent()

    with pytest.raises(
        SymptomInputError,
        match="string",
    ):
        agent.execute(
            make_task(),
            {"user_message": 123},
        )

    assert provider.calls == []


def test_rejects_blank_and_oversized_messages():
    agent, provider = make_agent(max_message_chars=5)

    with pytest.raises(SymptomInputError, match="blank"):
        agent.execute(
            make_task(),
            {"user_message": "   "},
        )

    with pytest.raises(SymptomInputError, match="length limit"):
        agent.execute(
            make_task(),
            {"user_message": "123456"},
        )

    assert provider.calls == []


def test_rejects_task_for_another_agent():
    agent, provider = make_agent()

    with pytest.raises(
        SymptomInputError,
        match="another agent",
    ):
        agent.execute(
            make_task(agent="medical_rag"),
            {"user_message": "Tôi bị đau đầu."},
        )

    assert provider.calls == []


def test_passes_generation_config_to_provider():
    config = GenerationConfig(
        temperature=0.2,
        num_ctx=8192,
        num_predict=512,
    )

    agent, provider = make_agent(
        generation_config=config,
    )

    agent.execute(
        make_task(),
        {"user_message": "Tôi bị nổi mẩn đỏ."},
    )

    assert provider.configs == [config]


def test_agent_runs_through_existing_orchestrator():
    agent, _ = make_agent()

    registry = AgentRegistry()
    registry.register(agent)

    orchestrator = Orchestrator(registry=registry)

    plan = {
        "intent": "symptom_question",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "symptom",
                "objective": "Extract explicitly reported symptoms.",
            }
        ],
    }

    results = orchestrator.execute_plan(
        plan,
        context={
            "request_id": "symptom-test-001",
            "user_message": "Tôi bị nổi mẩn đỏ.",
        },
    )

    assert len(results) == 1
    assert results[0].status == "succeeded"
    assert isinstance(
        results[0].output,
        SymptomExtractionOutput,
    )
def test_no_symptoms_rejects_unnecessary_follow_up_questions():
    agent, provider = make_agent(
        make_output(
            status="no_symptoms",
            follow_up_questions=[
                "Bạn có đang bị đau đầu không?"
            ],
        )
    )

    with pytest.raises(SymptomGenerationError):
        agent.execute(
            make_task(),
            {"user_message": "Dị ứng latex là gì?"},
        )

    assert len(provider.calls) == 1
def test_extracts_multiple_symptoms_using_independent_exact_spans():
    message = "Mấy bữa nay tui bị nhức đầu với chóng mặt."

    generated = make_output(
        symptoms=[
            make_symptom(
                name="nhức đầu",
                reported_text="nhức đầu",
                polarity="present",
            ),
            make_symptom(
                name="chóng mặt",
                reported_text="chóng mặt",
                polarity="present",
            ),
        ]
    )

    agent, _ = make_agent(generated)

    output = agent.execute(
        make_task(),
        {"user_message": message},
    )

    assert len(output.symptoms) == 2

    assert output.symptoms[0].reported_text == "nhức đầu"
    assert output.symptoms[1].reported_text == "chóng mặt"

    for symptom in output.symptoms:
        assert symptom.reported_text in message
def test_temporal_schema_fields_have_distinct_descriptions():
    agent, _ = make_agent()

    properties = (
        agent.generation_schema[
            "$defs"
        ]["SymptomMention"]["properties"]
    )

    onset_description = properties["onset_text"]["description"]
    duration_description = properties["duration_text"]["description"]

    assert "when the symptom" in onset_description
    assert "how long the symptom" in duration_description
    assert onset_description != duration_description


def test_system_prompt_distinguishes_onset_and_duration():
    agent, _ = make_agent()

    prompt = agent._system_prompt

    assert "Onset vs. Duration" in prompt
    assert "Never copy the same phrase into both fields." in prompt
    assert (
        "If the message only states an elapsed duration, populate"
        in prompt
    )
def test_rejects_normalized_name_as_non_verbatim_reported_text():
    message = (
        "Mấy hôm nay em cứ buồn nôn, "
        "ăn gì cũng thấy khó chịu."
    )

    generated = make_output(
        symptoms=[
            make_symptom(
                name="khó chịu khi ăn",
                reported_text="khó chịu khi ăn",
                duration_text="Mấy hôm nay",
            )
        ]
    )

    agent, _ = make_agent(generated)

    with pytest.raises(
        SymptomGenerationError,
        match="reported_text",
    ):
        agent.execute(
            make_task(),
            {"user_message": message},
        )


def test_accepts_normalized_name_with_verbatim_reported_text():
    message = (
        "Mấy hôm nay em cứ buồn nôn, "
        "ăn gì cũng thấy khó chịu."
    )

    generated = make_output(
        symptoms=[
            make_symptom(
                name="khó chịu khi ăn",
                reported_text="ăn gì cũng thấy khó chịu",
                duration_text="Mấy hôm nay",
            )
        ]
    )

    agent, _ = make_agent(generated)

    output = agent.execute(
        make_task(),
        {"user_message": message},
    )

    symptom = output.symptoms[0]

    assert symptom.name == "khó chịu khi ăn"
    assert symptom.reported_text == "ăn gì cũng thấy khó chịu"
    assert symptom.duration_text == "Mấy hôm nay"
    assert symptom.onset_text is None