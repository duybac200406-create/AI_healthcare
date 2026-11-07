import json

import httpx
import pytest

from ai_healthcare.providers.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)
from ai_healthcare.providers.ollama_provider import (
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
)


def make_provider(handler):
    client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )

    provider = OllamaEmbeddingProvider(
        client=client,
        model="bge-m3",
    )

    return provider, client


def test_embed_many_returns_valid_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert request.method == "POST"
        assert request.url.path == "/api/embed"
        assert payload["model"] == "bge-m3"
        assert payload["truncate"] is False

        return httpx.Response(
            200,
            json={
                "model": "bge-m3",
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            },
        )

    provider, client = make_provider(handler)

    with client:
        vectors = provider.embed_many(
            ["Vietnamese medical text", "English medical text"]
        )

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]
    assert provider.dimensions == 3


def test_embed_returns_single_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1, 0.2, 0.3]],
            },
        )

    provider, client = make_provider(handler)

    with client:
        vector = provider.embed("Test medical text")

    assert vector == [0.1, 0.2, 0.3]


def test_empty_batch_is_rejected() -> None:
    provider, client = make_provider(
        lambda request: httpx.Response(200, json={})
    )

    with client:
        with pytest.raises(ValueError, match="at least one"):
            provider.embed_many([])


def test_invalid_text_is_rejected() -> None:
    provider, client = make_provider(
        lambda request: httpx.Response(200, json={})
    )

    with client:
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed_many(["valid text", "   "])


def test_embedding_count_mismatch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1, 0.2]],
            },
        )

    provider, client = make_provider(handler)

    with client:
        with pytest.raises(
            OllamaResponseError,
            match="count does not match",
        ):
            provider.embed_many(["text one", "text two"])


def test_inconsistent_dimensions_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ],
            },
        )

    provider, client = make_provider(handler)

    with client:
        with pytest.raises(
            OllamaResponseError,
            match="inconsistent dimensions",
        ):
            provider.embed_many(["text one", "text two"])


def test_non_finite_values_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"embeddings":[[NaN]]}',
        )

    provider, client = make_provider(handler)

    with client:
        with pytest.raises(
            OllamaResponseError,
            match="non-finite",
        ):
            provider.embed("Test text")


def test_http_error_does_not_expose_response_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "Sensitive request information"},
        )

    provider, client = make_provider(handler)

    with client:
        with pytest.raises(OllamaHTTPError) as exc_info:
            provider.embed("Sensitive medical text")

    assert exc_info.value.status_code == 500
    assert "Sensitive request information" not in str(exc_info.value)


def test_timeout_is_handled_without_exposing_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Sensitive medical text"
        )

    provider, client = make_provider(handler)

    with client:
        with pytest.raises(OllamaTimeoutError) as exc_info:
            provider.embed("Sensitive medical text")

    assert "Sensitive medical text" not in str(exc_info.value)

def test_embedding_dimension_change_between_requests_is_rejected() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2, 0.3]]},
            )

        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2]]},
        )

    provider, client = make_provider(handler)

    with client:
        first_vector = provider.embed("First document")

        assert len(first_vector) == 3
        assert provider.dimensions == 3

        with pytest.raises(
            OllamaResponseError,
            match="dimension changed between requests",
        ):
            provider.embed("Second document")

    assert provider.dimensions == 3
    assert call_count == 2
