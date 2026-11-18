"""Deterministic sufficiency gate between the Symptom and Medical RAG agents.

The gate is NOT an agent and is never part of a Commander plan, so no model
can alter or bypass it. It is applied by wrapping the Medical RAG agent in
``GatedAgent`` when the service builds a turn.

Scope of Phase 4A: only fields that already exist on ``SymptomMention`` are
inspected, and clarification questions are generic templates. No
symptom-specific clinical content is written here; that must come from
sourced data (later phases).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.schemas.agent import AgentName, AgentTask
from ai_healthcare.schemas.conversation import (
    ClarificationNeeded,
    GateDecision,
)
from ai_healthcare.schemas.symptom import (
    SymptomExtractionOutput,
    SymptomMention,
)

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_NAME_CHARS = 60

ASK_DESCRIBE_SYMPTOM = (
    "Bạn đang gặp triệu chứng cụ thể nào? Hãy mô tả bằng lời của bạn."
)


@dataclass(frozen=True)
class GatePolicy:
    """Tunable thresholds. The defaults are a proposal, not a validated value."""

    max_clarification_rounds: int = 2
    require_temporal: bool = True  # onset_text or duration_text
    require_severity: bool = True
    max_questions: int = 3

    def __post_init__(self) -> None:
        if self.max_clarification_rounds < 0:
            raise ValueError("max_clarification_rounds must be >= 0.")
        if not 1 <= self.max_questions <= 3:
            raise ValueError("max_questions must be between 1 and 3.")


def _clean_name(name: str) -> str:
    cleaned = _CONTROL_RE.sub(" ", name).strip()
    return cleaned[:_MAX_NAME_CHARS].strip() or "triệu chứng này"


def _has_temporal(m: SymptomMention) -> bool:
    return bool(m.onset_text or m.duration_text)


def _is_sufficient(m: SymptomMention, policy: GatePolicy) -> bool:
    if policy.require_temporal and not _has_temporal(m):
        return False
    if policy.require_severity and not m.severity_text:
        return False
    return True


def _completeness(m: SymptomMention) -> int:
    return int(_has_temporal(m)) + int(bool(m.severity_text))


def _dedupe(questions: list[str], limit: int) -> tuple[str, ...]:
    seen: list[str] = []
    for q in questions:
        q = q.strip()
        if q and q not in seen:
            seen.append(q)
    return tuple(seen[:limit])


_TEMPORAL_QUESTION = "{Name} của bạn bắt đầu từ khi nào và đã kéo dài bao lâu?"
_SEVERITY_QUESTION = (
    "Mức độ {name} của bạn như thế nào (nhẹ, vừa hay nặng; "
    "có ảnh hưởng sinh hoạt không)?"
)


def evaluate_sufficiency(
    output: SymptomExtractionOutput,
    *,
    rounds: int,
    policy: GatePolicy = GatePolicy(),
    asked: frozenset[str] = frozenset(),
) -> GateDecision:
    """Decide whether Medical RAG may run for a symptom-originated question."""

    if rounds < 0:
        raise ValueError("rounds must be >= 0.")

    present = [m for m in output.symptoms if m.polarity == "present"]
    out_of_rounds = rounds >= policy.max_clarification_rounds

    # 1. The agent itself asked for clarification and rounds remain.
    if output.status == "needs_clarification" and not out_of_rounds:
        return GateDecision(
            allow=False,
            reason_code="needs_clarification_from_agent",
            questions=_dedupe(
                list(output.follow_up_questions), policy.max_questions
            ),
        )

    # 2. Nothing to retrieve for.
    if not present:
        if out_of_rounds:
            return GateDecision(
                allow=False, reason_code="no_symptom_identified"
            )
        return GateDecision(
            allow=False,
            reason_code="no_symptom_extracted",
            questions=(ASK_DESCRIBE_SYMPTOM,),
        )

    # 3. Enough information, whatever the round.
    if any(_is_sufficient(m, policy) for m in present):
        return GateDecision(allow=True, reason_code="sufficient")

    # 4. Something is still missing. Work out what, and what was never asked.
    target = max(present, key=_completeness)
    missing: list[str] = []
    if policy.require_temporal and not _has_temporal(target):
        missing.append("temporal")
    if policy.require_severity and not target.severity_text:
        missing.append("severity")
    unresolved = [a for a in missing if a not in asked]

    # 5. Never ask the same attribute twice: the user already answered once,
    #    even if the extractor could not capture it. Proceed, flagged limited.
    if out_of_rounds or not unresolved:
        return GateDecision(
            allow=True,
            reason_code=(
                "max_rounds_reached" if out_of_rounds else "already_asked"
            ),
            limited_information=True,
        )

    name = _clean_name(target.name)
    questions = [
        _TEMPORAL_QUESTION.format(Name=f"{name[:1].upper()}{name[1:]}")
        if attribute == "temporal"
        else _SEVERITY_QUESTION.format(name=name)
        for attribute in unresolved
    ]
    return GateDecision(
        allow=False,
        reason_code="insufficient_attributes",
        questions=_dedupe(questions, policy.max_questions),
        asked=tuple(unresolved),
    )


class GatedAgent(BaseAgent):
    """Decorator that applies the gate in front of the Medical RAG agent.

    One instance is created per turn. ``last_decision`` records what the gate
    decided so the service can read it without widening the orchestrator's
    context policy.
    """

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
        # Fail closed: an enforced gate with no symptom evidence must not
        # let the answer through.
        raise ValueError("Gate enforced but no symptom result was supplied.")
