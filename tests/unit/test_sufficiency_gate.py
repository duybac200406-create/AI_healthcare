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


def _sufficient(m):
    return bool(m.onset_text or m.duration_text) and bool(m.severity_text)


@settings(max_examples=300, deadline=None)
@given(outputs(), st.integers(min_value=0, max_value=5))
def test_gate_invariants(output, rounds):
    d = evaluate_sufficiency(output, rounds=rounds, policy=POLICY)
    present = [m for m in output.symptoms if m.polarity == "present"]

    # Deterministic.
    assert d == evaluate_sufficiency(output, rounds=rounds, policy=POLICY)
    # Question hygiene.
    assert len(d.questions) <= 3
    assert len(set(d.questions)) == len(d.questions)
    assert all(q.strip() for q in d.questions)
    if d.allow:
        assert d.questions == ()

    if rounds >= POLICY.max_clarification_rounds:
        assert d.allow == bool(present)
        enough = any(_sufficient(m) for m in present)
        # Limited only when the round budget, not the user, released it.
        assert d.limited_information == (d.allow and not enough)
        if d.allow and enough:
            assert d.reason_code == "sufficient"
    else:
        if d.allow:
            assert output.status != "needs_clarification"
            assert any(_sufficient(m) for m in present)
            assert not d.limited_information
        else:
            assert d.questions, "a blocked early round must ask something"


@settings(max_examples=300, deadline=None)
@given(outputs(), st.integers(min_value=0, max_value=1))
def test_adding_information_never_turns_allow_into_deny(output, rounds):
    before = evaluate_sufficiency(output, rounds=rounds, policy=POLICY)
    enriched = tuple(
        m.model_copy(
            update={
                "onset_text": m.onset_text or "x",
                "severity_text": m.severity_text or "y",
            }
        )
        for m in output.symptoms
    )
    after_output = SymptomExtractionOutput(
        status=output.status,
        symptoms=enriched,
        follow_up_questions=output.follow_up_questions,
    )
    after = evaluate_sufficiency(after_output, rounds=rounds, policy=POLICY)
    if before.allow:
        assert after.allow


class _CountingAgent(BaseAgent):
    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "medical_rag"

    def execute(self, task, context):
        self.calls += 1
        return "inner-output"


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


def _vague():
    return SymptomExtractionOutput(
        status="extracted",
        symptoms=(
            SymptomMention(
                name="a", reported_text="a", polarity="present",
                onset_text=None, duration_text=None,
                severity_text=None, body_site_text=None,
            ),
        ),
    )


def test_gated_agent_blocks_without_calling_inner():
    inner = _CountingAgent()
    agent = GatedAgent(inner, rounds=0)
    out = agent.execute(_task(), _context(_vague()))
    assert isinstance(out, ClarificationNeeded)
    assert inner.calls == 0
    assert agent.last_decision is not None and not agent.last_decision.allow


def test_gated_agent_passes_when_not_enforced():
    inner = _CountingAgent()
    out = GatedAgent(inner, rounds=0, enforce=False).execute(_task(), {})
    assert out == "inner-output" and inner.calls == 1


def test_gated_agent_fails_closed_without_symptom_result():
    inner = _CountingAgent()
    with pytest.raises(ValueError):
        GatedAgent(inner, rounds=0).execute(_task(), {})
    assert inner.calls == 0
