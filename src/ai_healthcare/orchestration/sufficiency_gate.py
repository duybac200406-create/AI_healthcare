"""Deterministic context gate between Symptom and Medical RAG.

This gate asks for one useful open-ended clarification when a symptom is
reported without any additional context. It does not require every symptom
field, and it is not a clinical triage or diagnostic rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.schemas.agent import AgentName, AgentTask
from ai_healthcare.schemas.conversation import ClarificationNeeded, GateDecision
from ai_healthcare.schemas.symptom import SymptomExtractionOutput, SymptomMention

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_NAME_CHARS = 60

ASK_DESCRIBE_SYMPTOM = (
    "Bạn đang gặp triệu chứng cụ thể nào? Hãy mô tả bằng lời của bạn."
)
ASK_FOR_CONTEXT = (
    "Bạn có thể mô tả thêm một vài chi tiết về triệu chứng không? "
    "Ví dụ: vị trí, cảm giác hoặc đặc điểm, diễn biến liên tục hay từng cơn, "
    "điều gì làm triệu chứng tăng hoặc giảm, triệu chứng đi kèm, hoặc ảnh "
    "hưởng đến sinh hoạt. Bạn chỉ cần chia sẻ điều mình biết; phần nào "
    "chưa rõ có thể bỏ qua."
)


@dataclass(frozen=True)
class GatePolicy:
    """Conversation policy, not a clinically validated sufficiency threshold."""

    max_clarification_rounds: int = 1
    require_context: bool = True
    max_questions: int = 1

    def __post_init__(self) -> None:
        if self.max_clarification_rounds < 0:
            raise ValueError("max_clarification_rounds must be >= 0.")
        if not 1 <= self.max_questions <= 3:
            raise ValueError("max_questions must be between 1 and 3.")


def _clean_name(name: str) -> str:
    cleaned = _CONTROL_RE.sub(" ", name).strip()
    return cleaned[:_MAX_NAME_CHARS].strip() or "triệu chứng này"


def _has_context(mention: SymptomMention) -> bool:
    """True when the user supplied any one useful descriptive attribute."""
    return any(
        value
        for value in (
            mention.onset_text,
            mention.duration_text,
            mention.severity_text,
            mention.body_site_text,
            mention.quality_text,
            mention.pattern_text,
            mention.aggravating_factors_text,
            mention.relieving_factors_text,
            mention.functional_impact_text,
        )
    )


def _dedupe(questions: list[str], limit: int) -> tuple[str, ...]:
    seen: list[str] = []
    for question in questions:
        cleaned = question.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen[:limit])


def evaluate_sufficiency(
    output: SymptomExtractionOutput,
    *,
    rounds: int,
    policy: GatePolicy = GatePolicy(),
    asked: frozenset[str] = frozenset(),
) -> GateDecision:
    """Decide whether the symptom-originated request can proceed to Medical RAG.

    One contextual detail is enough for the retrieval workflow. The gate does
    not infer medical safety or require onset/severity for every symptom.
    If the user cannot add details after one open question, RAG may proceed
    with limited-information disclosure instead of asking repeatedly.
    """
    if rounds < 0:
        raise ValueError("rounds must be >= 0.")

    present = [item for item in output.symptoms if item.polarity == "present"]
    out_of_rounds = rounds >= policy.max_clarification_rounds

    if output.status == "needs_clarification" and not out_of_rounds:
        return GateDecision(
            allow=False,
            reason_code="needs_clarification_from_agent",
            questions=_dedupe(
                list(output.follow_up_questions), policy.max_questions
            ),
        )

    if not present:
        if out_of_rounds:
            return GateDecision(
                allow=False,
                reason_code="no_symptom_identified",
            )
        return GateDecision(
            allow=False,
            reason_code="no_symptom_extracted",
            questions=(ASK_DESCRIBE_SYMPTOM,),
        )

    has_context = (
        not policy.require_context
        or len(present) > 1
        or any(_has_context(item) for item in present)
    )
    if has_context:
        return GateDecision(allow=True, reason_code="sufficient")

    if out_of_rounds or "context" in asked:
        return GateDecision(
            allow=True,
            reason_code=(
                "already_asked" if "context" in asked else "max_rounds_reached"
            ),
            limited_information=True,
        )

    name = _clean_name(present[0].name)
    question = ASK_FOR_CONTEXT.replace("triệu chứng", name, 1)
    return GateDecision(
        allow=False,
        reason_code="needs_context",
        questions=_dedupe([question], policy.max_questions),
        asked=("context",),
    )


class GatedAgent(BaseAgent):
    """Decorator that applies the deterministic context gate before Medical RAG."""

    def __init__(
        self,
        inner: BaseAgent,
        *,
        rounds: int,
        policy: GatePolicy = GatePolicy(),
        enforce: bool = True,
        asked: frozenset[str] = frozenset(),
    ) -> None:
        if not isinstance(inner, BaseAgent):
            raise TypeError("inner must be a BaseAgent.")
        self._inner = inner
        self._rounds = rounds
        self._policy = policy
        self._enforce = enforce
        self._asked = asked
        self.last_decision: GateDecision | None = None

    @property
    def name(self) -> AgentName:
        return self._inner.name

    def execute(self, task: AgentTask, context: dict[str, Any]) -> Any:
        if not self._enforce:
            return self._inner.execute(task, context)

        symptom_output = self._find_symptom_output(context)
        decision = evaluate_sufficiency(
            symptom_output,
            rounds=self._rounds,
            policy=self._policy,
            asked=self._asked,
        )
        self.last_decision = decision

        if not decision.allow:
            return ClarificationNeeded(decision=decision)
        return self._inner.execute(task, context)

    @staticmethod
    def _find_symptom_output(
        context: dict[str, Any],
    ) -> SymptomExtractionOutput:
        deps = context.get("dependency_results", {})
        for result in deps.values():
            if (
                result.agent == "symptom"
                and result.status == "succeeded"
                and isinstance(result.output, SymptomExtractionOutput)
            ):
                return result.output
        raise ValueError("Gate enforced but no symptom result was supplied.")
