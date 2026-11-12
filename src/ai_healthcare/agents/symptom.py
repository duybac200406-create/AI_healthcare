from __future__ import annotations
from ai_healthcare.agents.symptom_span_resolver import (
    resolve_present_carrier_spans,
)
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.providers.base import GenerationConfig, LLMProvider
from ai_healthcare.schemas.agent import AgentName, AgentTask
from ai_healthcare.schemas.symptom import SymptomExtractionOutput


DEFAULT_SYMPTOM_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts"
    / "symptom_extraction_system.md"
)


class SymptomAgentError(RuntimeError):
    """Base error raised by SymptomAgent."""


class SymptomInputError(SymptomAgentError):
    """Raised when the input violates the SymptomAgent contract."""


class SymptomGenerationError(SymptomAgentError):
    """Raised when model output violates the extraction contract."""


class SymptomAgent(BaseAgent):
    """Extract user-reported symptoms without making diagnoses."""

    _PATIENT_CONTEXT_KEYS = frozenset(
        {
            "patient_record",
            "patient_records",
            "patient_data",
            "patient_profile",
            "patient_context",
            "ehr",
            "emr",
            "ehr_record",
            "emr_record",
        }
    )

    def __init__(
        self,
        *,
        provider: LLMProvider,
        system_prompt: str | None = None,
        prompt_path: Path = DEFAULT_SYMPTOM_PROMPT_PATH,
        max_message_chars: int = 4000,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        model_name = getattr(provider, "model", None)

        if not isinstance(model_name, str) or not model_name.strip():
            raise TypeError("provider must expose a non-empty model name.")

        if not callable(getattr(provider, "generate_json", None)):
            raise TypeError("provider must implement generate_json().")

        if (
            isinstance(max_message_chars, bool)
            or not isinstance(max_message_chars, int)
        ):
            raise TypeError("max_message_chars must be an integer.")

        if max_message_chars < 1:
            raise ValueError("max_message_chars must be positive.")

        if generation_config is not None and not isinstance(
            generation_config,
            GenerationConfig,
        ):
            raise TypeError(
                "generation_config must be a GenerationConfig or None."
            )

        if system_prompt is not None and not isinstance(
            system_prompt,
            str,
        ):
            raise TypeError("system_prompt must be a string or None.")

        self._provider = provider
        self._model_name = model_name.strip()
        self._max_message_chars = max_message_chars
        self._generation_config = generation_config

        self._generation_schema = (
            SymptomExtractionOutput.model_json_schema()
        )

        base_prompt = (
            system_prompt
            if system_prompt is not None
            else prompt_path.read_text(encoding="utf-8")
        )

        if not base_prompt.strip():
            raise ValueError("Symptom system prompt must not be empty.")

        schema_text = json.dumps(
            self._generation_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        self._system_prompt = (
            f"{base_prompt.rstrip()}\n\n"
            "## Required SymptomExtractionOutput JSON Schema\n\n"
            f"{schema_text}\n"
        )

    @property
    def name(self) -> AgentName:
        return "symptom"

    @property
    def generation_schema(self) -> dict[str, Any]:
        return self._generation_schema

    def execute(
        self,
        task: AgentTask,
        context: dict[str, Any],
    ) -> SymptomExtractionOutput:
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask.")

        if task.agent != "symptom":
            raise SymptomInputError(
                "SymptomAgent received a task for another agent."
            )

        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary.")

        unsafe_keys = self._PATIENT_CONTEXT_KEYS.intersection(
            str(key).casefold() for key in context
        )

        if unsafe_keys:
            raise SymptomInputError(
                "SymptomAgent does not accept patient-record context."
            )

        message = context.get("user_message")

        if not isinstance(message, str):
            raise SymptomInputError(
                "context must contain a string 'user_message'."
            )

        message = message.strip()

        if not message:
            raise SymptomInputError("user_message must not be blank.")

        if len(message) > self._max_message_chars:
            raise SymptomInputError(
                "user_message exceeds the configured length limit."
            )

        user_prompt = json.dumps(
            {"user_message": message},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        raw_output = self._provider.generate_json(
            self._system_prompt,
            user_prompt,
            self._generation_schema,
            config=self._generation_config,
        )

        try:
            extraction = SymptomExtractionOutput.model_validate_json(
                raw_output
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise SymptomGenerationError(
                "Model output did not satisfy the SymptomExtractionOutput contract."
            ) from exc

               
        self._validate_source_spans(extraction, message)
        extraction = resolve_present_carrier_spans(
            extraction,
            message,
        )
        self._validate_source_spans(extraction, message)

        return extraction

    @staticmethod
    def _validate_source_spans(
        extraction: SymptomExtractionOutput,
        user_message: str,
    ) -> None:
        """Require every extracted source text field to exist in the input."""

        span_fields = (
            "reported_text",
            "onset_text",
            "duration_text",
            "severity_text",
            "body_site_text",
        )

        for index, symptom in enumerate(extraction.symptoms):
            for field_name in span_fields:
                span = getattr(symptom, field_name)

                if span is None:
                    continue

                if span not in user_message:
                    raise SymptomGenerationError(
                        "Model output contains a source span that "
                        f"does not match user_message "
                        f"(symptom index {index}, field {field_name})."
                    )
