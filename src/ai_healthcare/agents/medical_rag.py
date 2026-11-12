from __future__ import annotations
from ai_healthcare.providers.base import GenerationConfig, LLMProvider
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ai_healthcare.agents.base import BaseAgent
from ai_healthcare.schemas.agent import AgentName, AgentTask
from ai_healthcare.schemas.evidence import EvidenceReference
from ai_healthcare.schemas.medical_response import (
    GroundedClaim,
    MedicalGeneration,
    MedicalRAGOutput,
)
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorSearchResult


DEFAULT_MEDICAL_RAG_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "medical_rag_system.md"
)

class Retriever(Protocol):
    """Narrow retrieval contract consumed by this agent."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]: ...


class MedicalRAGError(RuntimeError):
    """Base error raised by MedicalRAGAgent."""


class MedicalRAGInputError(MedicalRAGError):
    """Raised when the task context does not satisfy the agent contract."""


class MedicalRAGGenerationError(MedicalRAGError):
    """Raised when the model response violates the generation contract."""


class MedicalRAGAgent(BaseAgent):
    """Answer general medical-information questions using retrieved evidence.

    The agent deliberately has no patient-record dependency. Its only input
    is context['user_message']; patient/EHR context keys are rejected.
    """

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
        retriever: Retriever,
        top_k: int = 5,
        score_threshold: float | None = None,
        system_prompt: str | None = None,
        prompt_path: Path = DEFAULT_MEDICAL_RAG_PROMPT_PATH,
        max_query_chars: int = 4000,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        model_name = getattr(provider, "model", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise TypeError("provider must expose a non-empty model name.")
        if not callable(getattr(provider, "generate_json", None)):
            raise TypeError("provider must implement generate_json().")
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must implement retrieve().")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise TypeError("score_threshold must be numeric.")
        if isinstance(max_query_chars, bool) or not isinstance(max_query_chars, int):
            raise TypeError("max_query_chars must be an integer.")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be positive.")
        if generation_config is not None and not isinstance(
            generation_config, GenerationConfig
        ):
            raise TypeError("generation_config must be a GenerationConfig or None.")
        self._provider = provider
        self._retriever = retriever
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._max_query_chars = max_query_chars
        self._generation_schema = MedicalGeneration.model_json_schema()
        self._generation_config = generation_config
        base_prompt = (
            system_prompt
            if system_prompt is not None
            else prompt_path.read_text(encoding="utf-8")
        )
        if not base_prompt.strip():
            raise ValueError("Medical RAG system prompt must not be empty.")
        schema_text = json.dumps(
            self._generation_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._system_prompt = (
            f"{base_prompt.rstrip()}\n\n"
            f"## Required MedicalGeneration JSON Schema\n\n{schema_text}\n"
        )

    @property
    def name(self) -> AgentName:
        return "medical_rag"

    @property
    def generation_schema(self) -> dict[str, Any]:
        return self._generation_schema

    def execute(self, task: AgentTask, context: dict[str, Any]) -> MedicalRAGOutput:
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask.")
        if task.agent != "medical_rag":
            raise MedicalRAGInputError("MedicalRAGAgent received a task for another agent.")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary.")

        unsafe_keys = self._PATIENT_CONTEXT_KEYS.intersection(
            str(key).casefold() for key in context
        )
        if unsafe_keys:
            raise MedicalRAGInputError(
                "MedicalRAGAgent does not accept patient-record context."
            )

        query = context.get("user_message")
        if not isinstance(query, str):
            raise MedicalRAGInputError("context must contain a string 'user_message'.")
        query = query.strip()
        if not query:
            raise MedicalRAGInputError("user_message must not be blank.")
        if len(query) > self._max_query_chars:
            raise MedicalRAGInputError("user_message exceeds the configured length limit.")

        retrieved = self._retriever.retrieve(
            query,
            top_k=self._top_k,
            filters=None,
            score_threshold=self._score_threshold,
        )
        if not isinstance(retrieved, Sequence) or isinstance(retrieved, (str, bytes)):
            raise MedicalRAGError("Retriever returned an invalid result collection.")
        if not all(isinstance(item, VectorSearchResult) for item in retrieved):
            raise MedicalRAGError("Retriever returned an invalid evidence item.")

        deduplicated: list[VectorSearchResult] = []
        seen: set[tuple[str, str, str]] = set()
        for item in retrieved:
            identity = (
                item.document_id,
                item.chunk_id,
                item.pipeline_fingerprint,
            )
            if identity not in seen:
                seen.add(identity)
                deduplicated.append(item)

        evidence = tuple(
            EvidenceReference.from_search_result(f"E{index}", result)
            for index, result in enumerate(deduplicated, start=1)
        )

        if not evidence:
            return MedicalRAGOutput(
                status="no_evidence",
                retrieved_evidence=(),
                abstention_reason="no_evidence",
                generation_model=self._provider.model,
            )

        # Synthetic fixtures are plumbing-test data, not clinical evidence.
        # Keep them visible in diagnostics/provenance, but never let them
        # support a generated medical answer.
        generation_evidence = tuple(
            item for item in evidence if item.source_type != "synthetic_fixture"
        )
        if not generation_evidence:
            return MedicalRAGOutput(
                status="insufficient_evidence",
                retrieved_evidence=evidence,
                abstention_reason="insufficient_evidence",
                generation_model=self._provider.model,
            )

        user_payload = {
            "user_question": query,
            "retrieved_evidence": [
                item.to_prompt_payload() for item in generation_evidence
            ],
        }
        user_prompt = json.dumps(
            user_payload,
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
            generation = MedicalGeneration.model_validate_json(raw_output)
        except (ValidationError, ValueError, TypeError) as exc:
            raise MedicalRAGGenerationError(
                "Model output did not satisfy the MedicalGeneration contract."
            ) from exc

        if generation.status == "insufficient_evidence":
            return MedicalRAGOutput(
                status="insufficient_evidence",
                retrieved_evidence=evidence,
                abstention_reason="insufficient_evidence",
                generation_model=self._provider.model,
            )

        if generation.status == "conflicting_evidence":
            return MedicalRAGOutput(
                status="conflicting_evidence",
                retrieved_evidence=evidence,
                abstention_reason="conflicting_evidence",
                generation_model=self._provider.model,
            )

        evidence_by_id = {item.evidence_id: item for item in generation_evidence}
        used_ids = {
            evidence_id
            for claim in generation.claims
            for evidence_id in claim.evidence_ids
        }
        if not used_ids.issubset(evidence_by_id):
            # Fail closed: never pass through a claim that cites an ID not
            # assigned by this exact retrieval call.
            return MedicalRAGOutput(
                status="generation_rejected",
                retrieved_evidence=evidence,
                abstention_reason="invalid_citations",
                generation_model=self._provider.model,
            )

        citations = tuple(
            item for item in generation_evidence if item.evidence_id in used_ids
        )
        claims = tuple(
            GroundedClaim(text=claim.text, evidence_ids=claim.evidence_ids)
            for claim in generation.claims
        )
        return MedicalRAGOutput(
            status="answered",
            claims=claims,
            citations=citations,
            retrieved_evidence=evidence,
            generation_model=self._provider.model,
        )
