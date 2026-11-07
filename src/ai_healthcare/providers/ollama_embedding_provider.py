from __future__ import annotations

import math
from collections.abc import Sequence
from types import TracebackType

import httpx

from ai_healthcare.providers.ollama_provider import (
    DEFAULT_BASE_URL,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
)


DEFAULT_EMBEDDING_MODEL = "bge-m3"


class OllamaEmbeddingProvider:
    """Generate text embeddings through a local Ollama server."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 120.0,
        truncate: bool = False,
        keep_alive: str | int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty.")

        if not model.strip():
            raise ValueError("model must not be empty.")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.truncate = truncate
        self.keep_alive = keep_alive

        self._owns_client = client is None
        self._client = (
            client if client is not None
            else httpx.Client(timeout=timeout)
        )

        self._dimensions: int | None = None

    @property
    def embed_url(self) -> str:
        """Return the Ollama embedding endpoint."""
        return f"{self.base_url}/api/embed"

    @property
    def dimensions(self) -> int | None:
        """Return the validated embedding dimension, if available."""
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for one text."""

        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

        if isinstance(texts, (str, bytes)) or not isinstance(
            texts, Sequence
        ):
            raise TypeError("texts must be a sequence of strings.")

        if not texts:
            raise ValueError("texts must contain at least one item.")

        if not all(
            isinstance(text, str) and text.strip()
            for text in texts
        ):
            raise ValueError(
                "Every input text must be a non-empty string."
            )

        payload = {
            "model": self.model,
            "input": list(texts),
            "truncate": self.truncate,
        }

        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        try:
            response = self._client.post(
                self.embed_url,
                json=payload,
            )

        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama embedding request timed out after "
                f"{self.timeout}s."
            ) from exc

        except httpx.TransportError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}: "
                f"{type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            # Do not expose request text or raw response details.
            raise OllamaHTTPError(response.status_code)

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama embedding response is not valid JSON."
            ) from exc

        return self._validate_embeddings(
            data,
            expected_count=len(texts),
        )

    def _validate_embeddings(
        self,
        data: object,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate the structure and numerical values of embeddings."""

        if not isinstance(data, dict):
            raise OllamaResponseError(
                "Ollama embedding response must be a JSON object."
            )

        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise OllamaResponseError(
                "Ollama response is missing 'embeddings'."
            )

        if len(embeddings) != expected_count:
            raise OllamaResponseError(
                "Embedding count does not match the input count."
            )

        validated: list[list[float]] = []

        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise OllamaResponseError(
                    "Embedding vector must be a non-empty list."
                )

            values: list[float] = []

            for value in vector:
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise OllamaResponseError(
                        "Embedding vector contains a non-numeric value."
                    )

                number = float(value)

                if not math.isfinite(number):
                    raise OllamaResponseError(
                        "Embedding vector contains a non-finite value."
                    )

                values.append(number)

            validated.append(values)

        dimensions = len(validated[0])

        if any(
            len(vector) != dimensions
            for vector in validated
        ):
            raise OllamaResponseError(
                "Embedding vectors have inconsistent dimensions."
            )

        if (
            self._dimensions is not None
            and dimensions != self._dimensions
        ):
            raise OllamaResponseError(
                "Embedding dimension changed between requests."
            )

        self._dimensions = dimensions

        return validated

    def close(self) -> None:
        """Close the HTTP client only when owned by this provider."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaEmbeddingProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
