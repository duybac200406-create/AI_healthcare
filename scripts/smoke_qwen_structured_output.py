"""Verify Qwen/Ollama JSON-schema output without using medical evidence.

This is a provider connectivity/structured-output smoke test only. It does not
validate MedicalRAG answer quality or clinical grounding.
"""

from __future__ import annotations

import json

from ai_healthcare.providers.ollama_provider import OllamaProvider


SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "message": {"type": "string", "minLength": 1},
    },
    "required": ["status", "message"],
    "additionalProperties": False,
}


def main() -> None:
    with OllamaProvider(model="qwen3:8b", temperature=0, num_predict=96) as provider:
        raw = provider.generate_json(
            system_prompt=(
                "You are testing JSON structured output. Do not provide medical advice. "
                "Return status ok and a short diagnostic message."
            ),
            user_prompt="Return the required JSON object only.",
            json_schema=SCHEMA,
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Qwen returned invalid JSON during the smoke test.") from exc

    if not isinstance(data, dict) or data.get("status") != "ok":
        raise RuntimeError("Qwen response did not satisfy the smoke-test contract.")
    if not isinstance(data.get("message"), str) or not data["message"].strip():
        raise RuntimeError("Qwen response is missing its diagnostic message.")

    print("MODEL: qwen3:8b")
    print("STRUCTURED OUTPUT: PASS")
    print("MESSAGE:", data["message"])


if __name__ == "__main__":
    main()
