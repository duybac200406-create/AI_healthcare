import pytest
from hypothesis import given, settings, strategies as st

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.orchestration.sufficiency_gate import (
    GatePolicy,
    GatedAgent,
    evaluate_sufficiency,
)
from ai_healthcare.schemas.agent import AgentTask
from ai_healthcare.schemas.conversation import ClarificationNeeded
from ai_healthcare.schemas.execution import AgentExecutionResult
from ai_healthcare.schemas.symptom import (
    SymptomExtractionOutput,
    SymptomMention,
)

POLICY = GatePolicy()
_word = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=12,
)
_optional = st.one_of(st.none(), _word)


@st.composite
def mentions(draw):
    return SymptomMention(
        name=draw(_word),
        reported_text=draw(_word),
        polarity=draw(st.sampled_from(["present", "absent", "uncertain"])),
        onset_text=draw(_optional),
        duration_text=draw(_optional),
        severity_text=draw(_optional),
        body_site_text=draw(_optional),
        quality_text=draw(_optional),
        pattern_text=draw(_optional),
        aggravating_factors_text=draw(_optional),
        relieving_factors_text=draw(_optional),
        functional_impact_text=draw(_optional),
    )


@st.composite
def outputs(draw):
    status = draw(
        st.sampled_from(["extracted", "no_symptoms", "needs_clarification"])
    )
    items = ()
    if status != "no_symptoms":
        items = tuple(
            draw(
                st.lists(
                    mentions(),
                    min_size=1 if status == "extracted" else 0,
                    max_size=4,
                    unique_by=lambda m: (
                        m.name.casefold(),
                        m.reported_text.casefold(),
                        m.polarity,
                    ),
                )
            )
        )
    follow_ups = ()
    if status == "needs_clarification":
        follow_ups = tuple(
            draw(st.lists(_word, min_size=1, max_size=5))
        )
    return SymptomExtractionOutput(
        status=status, symptoms=items, follow_up_questions=follow_ups
    )


_CONTEXT_FIELDS = (
    "onset_text",
    "duration_text",
    "severity_text",
    "body_site_text",
    "quality_text",
    "pattern_text",
    "aggravating_factors_text",
    "relieving_factors_text",
    "functional_impact_text",
)


def _has_context(mention):
    return any(getattr(mention, field) for field in _CONTEXT_FIELDS)


def _task():
    return AgentTask(
        task_id="task_2",
        agent="medical_rag",
        objective="Answer with evidence.",
        depends_on=("task_1",),
    )


def _context(output):
    return {
        "dependency_results": {
            "task_1": AgentExecutionResult(
                task_id="task_1",
                agent="symptom",
                status="succeeded",
                output=output,
            )
        }
    }


def _mention(**updates):
    values = {
        "name": "đau bụng",
        "reported_text": "đau bụng",
        "polarity": "present",
        **{field: None for field in _CONTEXT_FIELDS},
    }
    values.update(updates)
    return SymptomMention(**values)


@settings(max_examples=400, deadline=None)
@given(
    outputs(),
    st.integers(min_value=0, max_value=4),
    st.frozensets(st.sampled_from(["context", "temporal", "severity"])),
)
def test_gate_invariants(output, rounds, asked):
    decision = evaluate_sufficiency(
        output, rounds=rounds, policy=POLICY, asked=asked
    )
    repeated = evaluate_sufficiency(
        output, rounds=rounds, policy=POLICY, asked=asked
    )
    assert decision == repeated
    assert len(decision.questions) <= 1
    assert len(decision.questions) == len(set(decision.questions))
    assert all(question.strip() for question in decision.questions)

    present = [m for m in output.symptoms if m.polarity == "present"]
    out_of_rounds = rounds >= POLICY.max_clarification_rounds

    if output.status == "needs_clarification" and not out_of_rounds:
        assert not decision.allow
        assert decision.reason_code == "needs_clarification_from_agent"
        assert decision.questions
    elif not present:
        assert not decision.allow
        if out_of_rounds:
            assert decision.reason_code == "no_symptom_identified"
            assert not decision.questions
        else:
            assert decision.reason_code == "no_symptom_extracted"
            assert decision.questions
    elif len(present) > 1 or any(_has_context(m) for m in present):
        assert decision.allow
        assert decision.reason_code == "sufficient"
        assert not decision.limited_information
        assert not decision.questions
    elif out_of_rounds or "context" in asked:
        assert decision.allow
        assert decision.limited_information
        assert decision.reason_code in {"already_asked", "max_rounds_reached"}
        assert not decision.questions
    else:
        assert not decision.allow
        assert decision.reason_code == "needs_context"
        assert len(decision.questions) == 1
        assert decision.asked == ("context",)


@settings(max_examples=300, deadline=None)
@given(
    outputs(),
    st.integers(min_value=0, max_value=1),
    st.frozensets(st.sampled_from(["context", "temporal", "severity"])),
)
def test_adding_context_never_turns_an_allowed_decision_into_a_block(output, rounds, asked):
    before = evaluate_sufficiency(
        output, rounds=rounds, policy=POLICY, asked=asked
    )
    enriched = tuple(
        m.model_copy(update={"quality_text": m.quality_text or "sharp"})
        for m in output.symptoms
    )
    after_output = SymptomExtractionOutput(
        status=output.status,
        symptoms=enriched,
        follow_up_questions=output.follow_up_questions,
    )
    after = evaluate_sufficiency(
        after_output, rounds=rounds, policy=POLICY, asked=asked
    )
    if before.allow:
        assert after.allow


def test_vague_symptom_gets_one_open_ended_context_question():
    output = SymptomExtractionOutput(
        status="extracted",
        symptoms=(_mention(),),
    )
    decision = evaluate_sufficiency(output, rounds=0)

    assert not decision.allow
    assert decision.reason_code == "needs_context"
    assert len(decision.questions) == 1
    assert "vị trí" in decision.questions[0]
    assert "đặc điểm" in decision.questions[0]
    assert decision.asked == ("context",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("onset_text", "từ sáng nay"),
        ("duration_text", "ba ngày nay"),
        ("severity_text", "đau nặng"),
        ("body_site_text", "vùng bụng dưới"),
        ("quality_text", "âm ỉ"),
        ("pattern_text", "từng cơn"),
        ("aggravating_factors_text", "đau tăng khi đi lại"),
        ("relieving_factors_text", "nghỉ thì đỡ"),
        ("functional_impact_text", "khó ngủ vì đau"),
    ],
)
def test_any_one_explicit_context_attribute_allows_general_rag(field, value):
    output = SymptomExtractionOutput(
        status="extracted",
        symptoms=(_mention(**{field: value}),),
    )
    decision = evaluate_sufficiency(output, rounds=0)

    assert decision.allow
    assert decision.reason_code == "sufficient"
    assert not decision.limited_information


def test_an_additional_present_symptom_counts_as_context():
    output = SymptomExtractionOutput(
        status="extracted",
        symptoms=(
            _mention(),
            _mention(
                name="buồn nôn",
                reported_text="buồn nôn",
            ),
        ),
    )
    decision = evaluate_sufficiency(output, rounds=0)
    assert decision.allow
    assert decision.reason_code == "sufficient"


def test_no_new_detail_after_open_question_proceeds_with_limited_information():
    output = SymptomExtractionOutput(
        status="extracted",
        symptoms=(_mention(),),
    )
    decision = evaluate_sufficiency(
        output,
        rounds=1,
        asked=frozenset({"context"}),
    )
    assert decision.allow
    assert decision.reason_code == "already_asked"
    assert decision.limited_information


def test_no_symptom_after_round_limit_does_not_call_rag():
    output = SymptomExtractionOutput(status="no_symptoms")
    decision = evaluate_sufficiency(output, rounds=1)
    assert not decision.allow
    assert decision.reason_code == "no_symptom_identified"


class _CountingAgent(BaseAgent):
    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "medical_rag"

    def execute(self, task, context):
        self.calls += 1
        return "inner-output"


def test_gated_agent_blocks_without_any_context():
    inner = _CountingAgent()
    agent = GatedAgent(inner, rounds=0)
    out = agent.execute(_task(), _context(
        SymptomExtractionOutput(
            status="extracted",
            symptoms=(_mention(),),
        )
    ))
    assert isinstance(out, ClarificationNeeded)
    assert inner.calls == 0
    assert agent.last_decision is not None
    assert agent.last_decision.reason_code == "needs_context"


def test_gated_agent_passes_when_one_context_field_exists():
    inner = _CountingAgent()
    out = GatedAgent(inner, rounds=0).execute(
        _task(),
        _context(
            SymptomExtractionOutput(
                status="extracted",
                symptoms=(_mention(pattern_text="từng cơn"),),
            )
        ),
    )
    assert out == "inner-output"
    assert inner.calls == 1


def test_gated_agent_passes_when_not_enforced():
    inner = _CountingAgent()
    out = GatedAgent(inner, rounds=0, enforce=False).execute(_task(), {})
    assert out == "inner-output"
    assert inner.calls == 1


def test_gated_agent_fails_closed_without_symptom_result():
    inner = _CountingAgent()
    with pytest.raises(ValueError, match="no symptom result"):
        GatedAgent(inner, rounds=0).execute(_task(), {})
    assert inner.calls == 0
