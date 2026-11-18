"""Multi-turn conversation service (Phase 4A).

Responsibilities
----------------
* normalize input once (NFC) and apply an accent-insensitive record guard;
* route through the Commander, or resume an open clarification episode;
* expand a symptom-only plan into ``symptom -> medical_rag`` behind the
  deterministic sufficiency gate;
* run ``medical_information`` and ``mixed`` plans unchanged (ungated);
* return a serializable ``TurnResult`` and the next ``ConversationState``.

The Commander, orchestrator and agents are not modified. Logging records
reason codes and statuses only, never user or medical text.
"""

from __future__ import annotations

import logging
import time
import uuid

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.agents.commander import (
    Commander,
    CommanderError,
    CommanderPolicyError,
)
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.orchestration.response_assembler import ResponseAssembler
from ai_healthcare.orchestration.language_guard import LanguageGuardedAgent
from ai_healthcare.orchestration.sufficiency_gate import GatedAgent, GatePolicy
from ai_healthcare.schemas.agent import AgentTask, CommanderPlan
from ai_healthcare.schemas.conversation import (
    ClarificationNeeded,
    ConversationState,
    GateDecision,
    Turn,
    TurnKind,
    TurnResult,
)
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.schemas.symptom import SymptomExtractionOutput
from ai_healthcare.services.input_guard import (
    MAX_USER_MESSAGE_CHARS,
    is_patient_record_request,
    normalize_user_text,
)

logger = logging.getLogger(__name__)

MSG_EMPTY = "Vui lòng nhập nội dung câu hỏi."
MSG_TOO_LONG = (
    "Nội dung hội thoại hiện tại quá dài. Hãy bắt đầu một chủ đề mới."
)
MSG_UNSUPPORTED = (
    "Yêu cầu này nằm ngoài phạm vi hiện tại của hệ thống demo "
    "(thông tin y khoa chung và mô tả triệu chứng)."
)
MSG_RECORD = (
    "Hệ thống demo không truy cập hồ sơ cá nhân hay hồ sơ bệnh án."
)
MSG_NO_SYMPTOM = (
    "Mình chưa xác định được triệu chứng cụ thể từ nội dung bạn cung cấp. "
    "Bạn có thể mô tả lại triệu chứng hoặc đặt một câu hỏi y khoa chung."
)
MSG_ERROR = "Hệ thống gặp lỗi khi xử lý yêu cầu. Vui lòng thử lại."
MSG_LANGUAGE = (
    "Tài liệu nguồn bằng tiếng Anh; một phần câu trả lời có thể hiển thị "
    "bằng tiếng Anh."
)

_SYMPTOM_TASK = AgentTask(
    task_id="task_1",
    agent="symptom",
    objective="Extract the symptoms stated by the user.",
)
_RAG_TASK = AgentTask(
    task_id="task_2",
    agent="medical_rag",
    objective="Provide general medical information using retrieved evidence.",
    depends_on=("task_1",),
)


def symptom_workflow_plan() -> CommanderPlan:
    """Deterministic expansion of a symptom-only request."""

    return CommanderPlan(
        intent="mixed",
        status="ready",
        tasks=(_SYMPTOM_TASK, _RAG_TASK),
    )


def build_coverage_notice(
    output: MedicalRAGOutput,
    *,
    limited_information: bool,
) -> str | None:
    """Neutral disclosure built from the metadata of the evidence used."""

    evidence = output.citations or output.retrieved_evidence
    if not evidence:
        return None

    titles = list(dict.fromkeys(e.title for e in evidence))
    sources = list(dict.fromkeys(e.source_name for e in evidence))
    notice = (
        f"Nguồn sử dụng: {', '.join(titles)} ({'; '.join(sources)}). "
        "Đây là tóm tắt theo chủ đề, không cho biết nguyên nhân của "
        "triệu chứng của bạn."
    )
    if limited_information:
        notice += (
            " Thông tin về triệu chứng của bạn còn hạn chế nên nội dung "
            "chỉ mang tính tham khảo chung."
        )
    return notice


class ConversationService:
    """Stateless service: all conversation state travels in/out explicitly."""

    def __init__(
        self,
        *,
        commander: Commander,
        symptom_agent: BaseAgent,
        medical_rag_agent: BaseAgent,
        policy: GatePolicy | None = None,
        assembler: ResponseAssembler | None = None,
    ) -> None:
        self._commander = commander
        self._symptom_agent = symptom_agent
        self._rag_agent = medical_rag_agent
        self._policy = policy or GatePolicy()
        self._assembler = assembler or ResponseAssembler()

    # ------------------------------------------------------------------ API

    def handle_turn(
        self,
        state: ConversationState,
        message: str,
    ) -> TurnResult:
        started = time.perf_counter()
        result = self._handle(state, message)
        elapsed = int((time.perf_counter() - started) * 1000)
        result = result.model_copy(update={"latency_ms": elapsed})
        logger.info(
            "turn kind=%s phase=%s->%s gate=%s ms=%d",
            result.kind,
            state.phase,
            result.state.phase,
            result.gate.reason_code if result.gate else None,
            elapsed,
        )
        return result

    # ------------------------------------------------------------- internals

    def _handle(self, state: ConversationState, message: str) -> TurnResult:
        text = normalize_user_text(message)

        if not text:
            return self._terminal(state, "error", MSG_EMPTY, user_text=None)

        if is_patient_record_request(text):
            return self._terminal(state, "unsupported", MSG_RECORD, text)

        episode = state.episode_user_texts
        resumed_symptom = state.phase == "awaiting_symptom_details"
        resumed_routing = state.phase == "awaiting_routing_clarification"

        effective = "\n".join((*episode, text)) if episode else text
        if len(effective) > MAX_USER_MESSAGE_CHARS:
            return self._terminal(state, "error", MSG_TOO_LONG, text, reset=True)

        proposed: CommanderPlan | None = None
        if resumed_symptom:
            plan, gated = symptom_workflow_plan(), True
        else:
            try:
                proposed = self._commander.plan(effective)
            except CommanderPolicyError:
                return self._terminal(
                    state, "unsupported", MSG_UNSUPPORTED, text, reset=True
                )
            except CommanderError:
                return self._terminal(state, "error", MSG_ERROR, text)
            except Exception as exc:  # provider/network failures
                logger.warning("commander failed: %s", type(exc).__name__)
                return self._terminal(state, "error", MSG_ERROR, text)

            if proposed.status == "unsupported":
                return self._terminal(
                    state, "unsupported", MSG_UNSUPPORTED, text,
                    reset=True, proposed_plan=proposed,
                )
            if proposed.status == "needs_clarification":
                question = proposed.clarification_question or MSG_UNSUPPORTED
                return self._open_episode(
                    state, "awaiting_routing_clarification",
                    texts=(*episode, text), questions=(question,),
                    rounds=0, user_text=text, proposed_plan=proposed,
                )

            agents = {t.agent for t in proposed.tasks}
            if agents == {"symptom"}:
                plan, gated = symptom_workflow_plan(), True
            else:
                plan, gated = proposed, False

        return self._execute(
            state, text, effective, plan, gated, proposed,
            episode=(*episode, text),
            rounds=state.clarification_rounds if resumed_symptom else 0,
            resumed_routing=resumed_routing,
        )

    def _execute(
        self,
        state: ConversationState,
        text: str,
        effective: str,
        plan: CommanderPlan,
        gated: bool,
        proposed: CommanderPlan | None,
        *,
        episode: tuple[str, ...],
        rounds: int,
        resumed_routing: bool,
    ) -> TurnResult:
        language_agent = LanguageGuardedAgent(
            self._rag_agent, user_message=effective
        )
        gate_agent = GatedAgent(
            language_agent, rounds=rounds, policy=self._policy, enforce=gated
        )
        registry = AgentRegistry()
        registry.register(self._symptom_agent)
        registry.register(gate_agent)
        orchestrator = Orchestrator(
            enabled_agents={"symptom", "medical_rag"}, registry=registry
        )

        try:
            results = orchestrator.execute_plan(
                plan,
                context={
                    "request_id": uuid.uuid4().hex,
                    "user_message": effective,
                },
            )
            assembled = self._assembler.assemble(results)
        except Exception as exc:
            logger.warning("execution failed: %s", type(exc).__name__)
            return self._terminal(state, "error", MSG_ERROR, text, proposed_plan=proposed)

        by_agent = {r.agent: r for r in results}
        sym = by_agent.get("symptom")
        rag = by_agent.get("medical_rag")

        symptom_output = (
            sym.output
            if sym is not None
            and sym.status == "succeeded"
            and isinstance(sym.output, SymptomExtractionOutput)
            else None
        )
        decision: GateDecision | None = gate_agent.last_decision
        language_outcome = language_agent.outcome
        common = dict(
            proposed_plan=proposed,
            executed_plan=plan,
            assembled=assembled,
            gate=decision,
            symptom_output=symptom_output,
        )

        if rag is not None and rag.status == "succeeded":
            if isinstance(rag.output, ClarificationNeeded):
                d = rag.output.decision
                if d.questions:
                    return self._open_episode(
                        state, "awaiting_symptom_details",
                        texts=episode, questions=d.questions,
                        rounds=rounds + 1, user_text=text, **common,
                    )
                return self._terminal(
                    state, "unsupported", MSG_NO_SYMPTOM, text, **common
                )

            if isinstance(rag.output, MedicalRAGOutput):
                notice = (
                    build_coverage_notice(
                        rag.output,
                        limited_information=bool(
                            decision and decision.limited_information
                        ),
                    )
                    if gated
                    else None
                )
                return self._answer(
                    state, text, rag.output, notice,
                    language_notice=(
                        MSG_LANGUAGE
                        if language_outcome == "retried_still_mismatch"
                        else None
                    ),
                    language_outcome=language_outcome,
                    **common,
                )

        # An agent failed, was skipped, or is not implemented.
        return self._terminal(state, "error", MSG_ERROR, text, **common)

    # ------------------------------------------------------ state builders

    @staticmethod
    def _append_turns(
        state: ConversationState, user_text: str | None, assistant_text: str | None
    ) -> tuple[Turn, ...]:
        turns = list(state.turns)
        if user_text:
            turns.append(Turn(role="user", text=user_text[:8000]))
        if assistant_text:
            turns.append(Turn(role="assistant", text=assistant_text[:8000]))
        return tuple(turns)

    def _terminal(
        self,
        state: ConversationState,
        kind: TurnKind,
        message: str,
        user_text: str | None,
        *,
        reset: bool = False,
        **extra,
    ) -> TurnResult:
        turns = self._append_turns(state, user_text, message)
        if reset or extra.get("executed_plan") is not None:
            new_state = ConversationState(
                conversation_id=state.conversation_id, turns=turns
            )
        else:
            new_state = state.model_copy(update={"turns": turns})
        return TurnResult(kind=kind, state=new_state, message=message, **extra)

    def _open_episode(
        self,
        state: ConversationState,
        phase: str,
        *,
        texts: tuple[str, ...],
        questions: tuple[str, ...],
        rounds: int,
        user_text: str,
        **extra,
    ) -> TurnResult:
        turns = self._append_turns(state, user_text, "\n".join(questions))
        new_state = ConversationState(
            conversation_id=state.conversation_id,
            phase=phase,
            turns=turns,
            episode_user_texts=texts,
            pending_questions=questions,
            clarification_rounds=rounds,
        )
        return TurnResult(
            kind="clarification",
            state=new_state,
            questions=questions,
            **extra,
        )

    def _answer(
        self,
        state: ConversationState,
        user_text: str,
        output: MedicalRAGOutput,
        notice: str | None,
        language_notice: str | None = None,
        language_outcome: str | None = None,
        **extra,
    ) -> TurnResult:
        shown = output.answer_text or "(Không có nội dung trả lời.)"
        turns = self._append_turns(state, user_text, shown)
        new_state = ConversationState(
            conversation_id=state.conversation_id, turns=turns
        )
        return TurnResult(
            kind="answer",
            state=new_state,
            medical_output=output,
            coverage_notice=notice,
            language_notice=language_notice,
            language_outcome=language_outcome,
            **extra,
        )
