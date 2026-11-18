"""Output-language guard for the Medical RAG agent.

The generation prompt asks for the user's language, but a model can still
copy English evidence verbatim. This decorator checks the result
deterministically and, when a Vietnamese question received a non-Vietnamese
answer, retries once with an explicit directive. It never alters evidence,
citations or the gate; it only decides whether to keep or replace an answer.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.schemas.agent import AgentName, AgentTask
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.services.language import detect_language

logger = logging.getLogger(__name__)

LanguageOutcome = Literal[
    "not_applicable", "ok", "retried_ok", "retried_still_mismatch"
]

RETRY_DIRECTIVE = (
    "\n(Hãy trả lời hoàn toàn bằng tiếng Việt; có thể giữ thuật ngữ chuyên "
    "môn tiếng Anh trong ngoặc đơn.)"
)


def _answer_language(output: MedicalRAGOutput) -> str:
    return detect_language(" ".join(claim.text for claim in output.claims))


class LanguageGuardedAgent(BaseAgent):
    """One instance per turn; ``outcome`` is read by the service."""

    def __init__(self, inner: BaseAgent, *, user_message: str) -> None:
        if not isinstance(inner, BaseAgent):
            raise TypeError("inner must be a BaseAgent.")
        self._inner = inner
        self._expects_vietnamese = detect_language(user_message) == "vi"
        self.outcome: LanguageOutcome = "not_applicable"

    @property
    def name(self) -> AgentName:
        return self._inner.name

    def execute(self, task: AgentTask, context: dict[str, Any]) -> Any:
        output = self._inner.execute(task, context)

        if (
            not self._expects_vietnamese
            or not isinstance(output, MedicalRAGOutput)
            or output.status != "answered"
        ):
            return output

        if _answer_language(output) != "other":
            self.outcome = "ok"
            return output

        retry_context = {
            **context,
            "user_message": f"{context['user_message']}{RETRY_DIRECTIVE}",
        }
        try:
            retry = self._inner.execute(task, retry_context)
        except Exception as exc:  # keep the grounded answer, disclose later
            logger.warning("language retry failed: %s", type(exc).__name__)
            self.outcome = "retried_still_mismatch"
            return output

        if (
            isinstance(retry, MedicalRAGOutput)
            and retry.status == "answered"
            and _answer_language(retry) != "other"
        ):
            self.outcome = "retried_ok"
            return retry

        self.outcome = "retried_still_mismatch"
        return output
