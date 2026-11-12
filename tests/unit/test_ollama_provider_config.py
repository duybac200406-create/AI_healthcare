import json

import httpx
import pytest

from ai_healthcare.providers.base import GenerationConfig
from ai_healthcare.providers.ollama_provider import OllamaProvider


SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


def test_generation_config_overrides_provider_defaults_per_request() -> None:
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": '{"answer": "ok"}'}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    provider = OllamaProvider(
        client=client,
        temperature=0.0,
        num_ctx=4096,
        num_predict=512,
    )

    try:
        custom_config = GenerationConfig(
            temperature=0.3,
            num_ctx=8192,
            num_predict=256,
        )

        provider.generate_json(
            "SYSTEM",
            "USER",
            SCHEMA,
            config=custom_config,
        )

        # A second call without config must use provider defaults.
        provider.generate_json("SYSTEM", "USER", SCHEMA)
    finally:
        client.close()

    custom_options = captured_payloads[0]["options"]
    default_options = captured_payloads[1]["options"]

    assert custom_options == {
        "temperature": 0.3,
        "num_ctx": 8192,
        "num_predict": 256,
    }

    assert default_options == {
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_predict": 512,
    }


def test_generation_config_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GenerationConfig(temperature=-0.1)


def test_generation_config_rejects_non_finite_temperature() -> None:
    with pytest.raises(ValueError, match="finite"):
        GenerationConfig(temperature=float("nan"))


def test_generation_config_rejects_invalid_context_length() -> None:
    with pytest.raises(ValueError, match="num_ctx must be positive"):
        GenerationConfig(num_ctx=0)


def test_generation_config_rejects_boolean_token_limit() -> None:
    with pytest.raises(TypeError, match="num_predict must be an integer"):
        GenerationConfig(num_predict=True)
