
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ai_healthcare.orchestration.orchestrator import (
    AgentNotEnabledError,
    Orchestrator,
)
from ai_healthcare.schemas.agent import CommanderPlan

# agents/commander.py -> agents -> ai_healthcare -> src -> <repository root>.
# Valid for the editable (src-layout) install this project uses.
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "commander_system.md"
)


class StructuredOutputProvider(Protocol):
    """Anything that can return a schema-constrained JSON string."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> str: ...


class CommanderError(Exception):
    """Base class for Commander errors."""


class CommanderOutputError(CommanderError):
    """The model answered, but its output did not become a valid plan.

    ``raw_output`` keeps the model text for debugging and evaluation. It is
    deliberately excluded from ``str(exc)`` so that it is not written to logs
    or tracebacks by accident.
    """

    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class CommanderInvalidJSONError(CommanderOutputError):
    """The model output is not parseable JSON."""


class CommanderSchemaError(CommanderOutputError):
    """The output is JSON but violates the ``CommanderPlan`` contract."""


class CommanderPolicyError(CommanderOutputError):
    """The plan is a valid ``CommanderPlan`` but the Orchestrator rejects it."""


class Commander:
    """Produce a validated ``CommanderPlan`` from a user message.

    Provider errors (``OllamaProviderError`` and subclasses) propagate
    unchanged: an infrastructure failure is a different event from a bad plan.
    """

    def __init__(
        self,
        provider: StructuredOutputProvider,
        *,
        orchestrator: Orchestrator | None = None,
        system_prompt: str | None = None,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._provider = provider
        self._orchestrator = orchestrator if orchestrator is not None else Orchestrator()
        self._json_schema: dict[str, Any] = CommanderPlan.model_json_schema()

        base_prompt = (
            system_prompt
            if system_prompt is not None
            else prompt_path.read_text(encoding="utf-8")
        )
        if not base_prompt.strip():
            raise ValueError("Commander system prompt must not be empty.")

        # The prompt refers to "the supplied CommanderPlan JSON Schema", so the
        # schema is appended verbatim. Routing rules are not modified.
        schema_text = json.dumps(
            self._json_schema, ensure_ascii=False, separators=(",", ":")
        )
        self._system_prompt = (
            f"{base_prompt.rstrip()}\n\n## CommanderPlan JSON Schema\n\n{schema_text}\n"
        )

    @property
    def json_schema(self) -> dict[str, Any]:
        """JSON Schema generated from ``CommanderPlan`` (a copy-safe reference)."""
        return self._json_schema

    @property
    def system_prompt(self) -> str:
        """Full system prompt sent to the model (rules + schema)."""
        return self._system_prompt

    
    @staticmethod
    def _is_patient_record_lookup(message: str) -> bool:
        """Detect explicit requests to access individual patient records.

        This guard must not block general educational questions
        about medical records or EHR/EMR systems.
        """
        text = " ".join(message.casefold().split())

        patterns = (
            # Vietnamese: an individual record is explicitly identified.
            r"\b(?:hồ sơ|bệnh án)\b.{0,100}"
            r"\b(?:của tôi|của mình|cá nhân|bệnh nhân)\b",

            # Vietnamese: a specific synthetic or patient record ID.
            r"\b(?:hồ sơ|bệnh án)\b.{0,60}"
            r"\b(?:syn[-\s]?\d+|[a-z]{2,}[-_]\d+)\b",

            # Vietnamese: explicit access action and individual target.
            r"\b(?:tra cứu|truy xuất|đọc|xem|mở|lấy|kiểm tra)"
            r"\b.{0,80}\b(?:hồ sơ|bệnh án)\b"
            r".{0,80}\b(?:của tôi|của mình|bệnh nhân)\b",

            # English: personal or patient-specific records.
            r"\b(?:my|personal|patient(?:'s)?)\s+"
            r"(?:medical\s+)?(?:record|records|chart|ehr|emr)\b",
        )

        return any(re.search(pattern, text) for pattern in patterns)

    def plan(self, user_message: str) -> CommanderPlan:
        """Return a validated plan for ``user_message``. Does not execute tasks.

        Raises:
            TypeError, ValueError: invalid ``user_message``.
            OllamaProviderError (or the provider's own errors): inference failed.
            CommanderInvalidJSONError: output is not JSON.
            CommanderSchemaError: output violates the ``CommanderPlan`` contract.
            CommanderPolicyError: the Orchestrator rejects the plan.
        """
        
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string.")
        message = user_message.strip()
        if not message:
            raise ValueError("user_message must not be empty.")
        if self._is_patient_record_lookup(message):
            blocked_plan = CommanderPlan(
                intent="unsupported",
                status="unsupported",
                tasks=(),
                clarification_question=None,
            )

            return self._orchestrator.validate_plan(blocked_plan)

        raw_output = self._provider.generate_json(
            self._system_prompt, message, self._json_schema
        )
        return self._validate(raw_output)

        
    
    @staticmethod
    def _normalize_plan_intent(data: dict[str, Any]) -> dict[str, Any]:

        status = data.get("status")
        if status == "unsupported":
            if (
                data.get("tasks") == []
                and "clarification_question" in data
                and data["clarification_question"] is None
            ):
                if data.get("intent") == "unsupported":
                    return data

                normalized_data = dict(data)
                normalized_data["intent"] = "unsupported"
                return normalized_data

            return data

        # Only normalize ready plans using selected agents.
        if status != "ready":
            return data

        tasks = data.get("tasks")

        if not isinstance(tasks, list):
            return data

        agents = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            agent = task.get("agent")

            if isinstance(agent, str) and agent not in agents:
                agents.append(agent)

        if len(agents) >= 2:
            normalized_intent = "mixed"

        elif len(agents) == 1:
            agent_to_intent = {
                "medical_rag": "medical_information",
                "symptom": "symptom_question",
                "medication": "medication_question",
                "psychology_support": "psychological_support",
            }

            normalized_intent = agent_to_intent.get(agents[0])

        else:
            return data

        if normalized_intent is None:
            return data

        if data.get("intent") == normalized_intent:
            return data

        normalized_data = dict(data)
        normalized_data["intent"] = normalized_intent

        return normalized_data

    def _validate(self, raw_output: str) -> CommanderPlan:
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CommanderInvalidJSONError(
                f"Model output is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}).",
                raw_output=raw_output,
            ) from exc

        if not isinstance(data, dict):
            raise CommanderSchemaError(
                "Model output must be a JSON object.", raw_output=raw_output
            )
        data = self._normalize_plan_intent(data)
        try:
            plan = CommanderPlan.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors(include_input=False, include_url=False)
            )
            raise CommanderSchemaError(
                f"Model output violates the CommanderPlan contract: {problems}",
                raw_output=raw_output,
            ) from exc

        try:
            return self._orchestrator.validate_plan(plan)
        except AgentNotEnabledError as exc:
            raise CommanderPolicyError(
                f"Orchestrator rejected the plan: {exc}", raw_output=raw_output
            ) from exc
