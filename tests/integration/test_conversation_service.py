"""Commander -> ConversationService -> Symptom + Medical RAG chain.

All models and the retriever are scripted stubs that react to neutral marker
tokens. These tests verify control flow, gating and state handling of
deterministic code. They make no statement about medical answer quality.
"""

import json
import unicodedata

from ai_healthcare.agents.commander import Commander
from ai_healthcare.agents.medical_rag import MedicalRAGAgent
from ai_healthcare.agents.symptom import SymptomAgent
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.schemas.conversation import ConversationState
from ai_healthcare.services.conversation_service import ConversationService
from tests.integration.test_medical_rag_orchestration import (
    FakeRetriever,
    make_search_result,
)


class Scripted:
    model = "stub"

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[str] = []

    def generate_json(self, system, user, schema, *, config=None):
        self.calls.append(user)
        return json.dumps(self.fn(user), ensure_ascii=False)


def _task(i, agent, deps=()):
    return {
        "task_id": f"task_{i}",
        "agent": agent,
        "objective": f"Do the {agent} work.",
        "depends_on": list(deps),
    }


def commander_fn(u):
    # Stand-in for a model that follows the user's latest reply.
    u = u.splitlines()[-1]

    def plan(intent, status, tasks=(), q=None):
        return {
            "intent": intent,
            "status": status,
            "tasks": list(tasks),
            "clarification_question": q,
        }

    if "[nope]" in u:
        return plan("unsupported", "unsupported")
    if "[unclear]" in u:
        return plan("symptom_question", "needs_clarification", q="Which topic?")
    if "[med]" in u:
        return plan("medication_question", "ready", [_task(1, "medication")])
    if "[both]" in u:
        return plan(
            "mixed", "ready", [_task(1, "symptom"), _task(2, "medical_rag")]
        )
    if "[info]" in u:
        return plan("medical_information", "ready", [_task(1, "medical_rag")])
    return plan("symptom_question", "ready", [_task(1, "symptom")])


def symptom_fn(u):
    if "symptom-X" not in u:
        return {"status": "no_symptoms", "symptoms": [], "follow_up_questions": []}
    return {
        "status": "extracted",
        "symptoms": [
            {
                "name": "symptom-X",
                "reported_text": "symptom-X",
                "polarity": "present",
                "onset_text": "Since yesterday" if "Since yesterday" in u else None,
                "duration_text": None,
                "severity_text": "severe" if "severe" in u else None,
                "body_site_text": None,
                "quality_text": None,
                "pattern_text": (
                    "comes in episodes" if "comes in episodes" in u else None
                ),
                "aggravating_factors_text": None,
                "relieving_factors_text": None,
                "functional_impact_text": (
                    "limits my daily activity"
                    if "limits my daily activity" in u else None
                ),
            }
        ],
        "follow_up_questions": [],
    }


RAG_OUT = {
    "status": "answered",
    "claims": [{"text": "Placeholder claim.", "evidence_ids": ["E1"]}],
    "reason": None,
}


class Rig:
    def __init__(self, symptom=symptom_fn, rag=None):
        self.cmd = Scripted(commander_fn)
        self.sym = Scripted(symptom)
        self.rag = Scripted(rag or (lambda u: RAG_OUT))
        self.retriever = FakeRetriever((make_search_result(),))
        sym_agent = SymptomAgent(provider=self.sym, system_prompt="T")
        rag_agent = MedicalRAGAgent(
            provider=self.rag, retriever=self.retriever, system_prompt="T"
        )
        registry = AgentRegistry()
        registry.register(sym_agent)
        registry.register(rag_agent)
        orchestrator = Orchestrator(
            enabled_agents={"symptom", "medical_rag"}, registry=registry
        )
        commander = Commander(
            self.cmd, orchestrator=orchestrator, system_prompt="T"
        )
        self.service = ConversationService(
            commander=commander,
            symptom_agent=sym_agent,
            medical_rag_agent=rag_agent,
        )
        self.state = ConversationState.new("c1")

    def say(self, message):
        result = self.service.handle_turn(self.state, message)
        self.state = result.state
        return result


def test_vague_symptom_gets_one_open_context_question_before_retrieval():
    rig = Rig()
    r = rig.say("I have symptom-X.")
    assert r.kind == "clarification"
    assert r.gate.reason_code == "needs_context"
    assert len(r.questions) == 1
    assert "vị trí" in r.questions[0]
    assert "diễn biến" in r.questions[0]
    assert rig.retriever.queries == [] and rig.rag.calls == []
    assert r.state.phase == "awaiting_symptom_details"
    assert r.state.clarification_rounds == 1
    assert r.state.asked_attributes == ("context",)


def test_replies_accumulate_and_skip_the_commander():
    rig = Rig()
    assert rig.say("I have symptom-X.").kind == "clarification"
    r = rig.say("Since yesterday and it is severe.")

    assert r.kind == "answer"
    assert r.gate.reason_code == "sufficient"
    assert r.coverage_notice
    assert len(rig.cmd.calls) == 1, "replies must not be re-routed"
    last_symptom_prompt = rig.sym.calls[-1]
    for fragment in ("symptom-X", "Since yesterday", "severe"):
        assert fragment in last_symptom_prompt
    assert len(rig.retriever.queries) == 1
    assert r.state.phase == "idle" and r.state.episode_user_texts == ()
    assert r.state.asked_attributes == ()


def test_user_who_cannot_add_more_detail_is_not_asked_again():
    rig = Rig()
    r1 = rig.say("I have symptom-X.")
    assert r1.kind == "clarification"
    assert r1.state.asked_attributes == ("context",)

    r2 = rig.say("I don't know.")
    assert r2.kind == "answer"
    assert r2.gate.reason_code == "already_asked"
    assert r2.gate.limited_information
    assert "hạn chế" in r2.coverage_notice


def test_alternative_context_detail_satisfies_gate_without_fixed_fields():
    rig = Rig()
    r1 = rig.say("I have symptom-X.")
    assert r1.kind == "clarification"
    assert r1.state.asked_attributes == ("context",)

    r2 = rig.say("It comes in episodes.")
    assert r2.kind == "answer"
    assert r2.gate.reason_code == "sufficient"
    assert r2.symptom_output.symptoms[0].pattern_text == "comes in episodes"


def test_agent_driven_clarification_does_not_loop_when_no_symptom_can_be_extracted():
    calls = {"n": 0}

    def asking_symptom(u):
        calls["n"] += 1
        return {
            "status": "needs_clarification",
            "symptoms": [],
            "follow_up_questions": ["Which symptom are you describing?"],
        }

    rig = Rig(symptom=asking_symptom)
    first = rig.say("I feel something unusual.")
    assert first.kind == "clarification"

    second = rig.say("I'm not sure.")
    assert second.kind == "unsupported"
    assert second.message
    assert calls["n"] == 2
    assert rig.retriever.queries == []


def test_no_symptom_after_round_limit_ends_the_episode():
    rig = Rig()
    assert rig.say("Random words only.").kind == "clarification"
    assert rig.say("Still nothing.").kind == "clarification"
    r = rig.say("Nothing.")
    assert r.kind == "unsupported"
    assert r.state.phase == "idle"
    assert rig.retriever.queries == []


def test_social_messages_get_a_friendly_reply_and_no_pipeline_call():
    rig = Rig()
    for message in ("xin chào", "cám ơn bạn", "tạm biệt"):
        r = rig.say(message)
        assert r.kind == "social"
        assert r.message
    assert rig.cmd.calls == [] and rig.sym.calls == []


def test_social_message_inside_an_episode_keeps_it_open():
    rig = Rig()
    r1 = rig.say("I have symptom-X.")
    assert r1.kind == "clarification"
    r2 = rig.say("xin chào")
    assert r2.kind == "social"
    assert r2.questions == r1.questions  # pending questions are repeated
    assert r2.state.phase == "awaiting_symptom_details"
    assert r2.state.clarification_rounds == r1.state.clarification_rounds
    assert len(rig.sym.calls) == 1  # the greeting was not analysed as a reply
    assert rig.say("Since yesterday and it is severe.").kind == "answer"


def test_medical_information_route_reaches_rag_directly():
    rig = Rig()
    r = rig.say("[info] general question")
    assert r.kind == "answer"
    assert {t.agent for t in r.executed_plan.tasks} == {"medical_rag"}
    assert rig.sym.calls == []
    assert r.gate is None and r.coverage_notice is None


def test_mixed_plan_is_executed_unchanged_and_ungated():
    rig = Rig()
    r = rig.say("[both] I have symptom-X, and a general question")
    assert r.kind == "answer"
    assert r.executed_plan == r.proposed_plan
    assert r.gate is None
    assert len(rig.retriever.queries) == 1


def test_unsupported_and_disabled_agent_paths():
    rig = Rig()
    assert rig.say("[nope] anything").kind == "unsupported"
    assert rig.say("[med] anything").kind == "unsupported"
    assert rig.retriever.queries == []


def test_routing_clarification_is_resumed_through_the_commander():
    rig = Rig()
    r1 = rig.say("[unclear] vague")
    assert r1.kind == "clarification"
    assert r1.state.phase == "awaiting_routing_clarification"
    r2 = rig.say("[info] details")
    assert r2.kind == "answer"
    assert len(rig.cmd.calls) == 2
    assert "[unclear] vague" in rig.cmd.calls[1]
    assert "[info] details" in rig.cmd.calls[1]


def test_patient_record_guard_runs_before_the_commander():
    rig = Rig()
    message = unicodedata.normalize("NFD", "Đọc hồ sơ cá nhân của tôi.")
    r = rig.say(message)
    assert r.kind == "unsupported"
    assert rig.cmd.calls == []


def test_decomposed_unicode_input_still_passes_span_validation():
    rig = Rig()
    message = unicodedata.normalize(
        "NFD", "Tôi có symptom-X. Since yesterday. It is severe."
    )
    r = rig.say(message)
    assert r.kind == "answer", r.message


def test_empty_message_keeps_an_open_episode():
    rig = Rig()
    assert rig.say("I have symptom-X.").kind == "clarification"
    r = rig.say("   ")
    assert r.kind == "error"
    assert r.state.phase == "awaiting_symptom_details"
    assert r.state.episode_user_texts == ("I have symptom-X.",)


def test_symptom_agent_failure_is_reported_and_rag_is_not_called():
    def boom(u):
        raise RuntimeError("provider down")

    rig = Rig(symptom=boom)
    r = rig.say("I have symptom-X.")
    assert r.kind == "error"
    assert r.state.phase == "idle"
    assert rig.retriever.queries == []


VI_ANSWER = {
    "status": "answered",
    "claims": [
        {"text": "Đây là câu trả lời bằng tiếng Việt.", "evidence_ids": ["E1"]}
    ],
    "reason": None,
}
VI_QUESTION = "[info] Cho mình hỏi thông tin chung về chủ đề này được không?"


def test_english_answer_to_a_vietnamese_question_is_retried_once():
    rig = Rig(rag=lambda u: VI_ANSWER if "tiếng Việt" in u.split("retrieved")[0] else RAG_OUT)
    r = rig.say(VI_QUESTION)
    assert r.kind == "answer"
    assert r.language_outcome == "retried_ok"
    assert r.language_notice is None
    assert len(rig.rag.calls) == 2
    assert "tiếng Việt" in r.medical_output.claims[0].text


def test_persistent_english_answer_is_disclosed_not_hidden():
    rig = Rig()  # always answers in English
    r = rig.say(VI_QUESTION)
    assert r.kind == "answer"
    assert r.language_outcome == "retried_still_mismatch"
    assert r.language_notice
    assert len(rig.rag.calls) == 2


def test_english_question_is_never_retried():
    rig = Rig()
    r = rig.say("[info] a general question written in English words")
    assert r.language_outcome == "not_applicable"
    assert len(rig.rag.calls) == 1
