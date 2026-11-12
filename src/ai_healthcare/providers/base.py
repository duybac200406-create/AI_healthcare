
"""Provider-neutral contracts for embedding and LLM generation backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Per-request generation options.

    None means that the provider's configured default is used.
    The object is immutable so concurrent agents cannot overwrite each
    other's generation settings.
    """

    temperature: float | None = None
    num_ctx: int | None = None
    num_predict: int | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(
                self.temperature, (int, float)
            ):
                raise TypeError("temperature must be a number or None.")
            if not isfinite(self.temperature):
                raise ValueError("temperature must be finite.")
            if self.temperature < 0:
                raise ValueError("temperature must be non-negative.")

        for field_name in ("num_ctx", "num_predict"):
            value = getattr(self, field_name)

            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{field_name} must be an integer or None.")
                if value <= 0:
                    raise ValueError(f"{field_name} must be positive.")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural interface implemented by embedding providers."""

    model: str

    @property
    def dimensions(self) -> int | None:
        """Validated embedding dimension, if the provider has generated one."""
        ...

    def embed(self, text: str) -> list[float]:
        """Generate one dense embedding."""
        ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface for schema-constrained LLM generation."""

    model: str

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a JSON string constrained by the supplied schema."""
        ...
