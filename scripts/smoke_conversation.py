"""Interactive terminal smoke test for the multi-turn pipeline.

Requires running Ollama and Qdrant. It never creates or modifies a
collection. Uses the same runtime builder as frontend/app.py.

Usage (PowerShell, repository root):
    python scripts/smoke_conversation.py
    python scripts/smoke_conversation.py --collection <existing collection>

Commands inside the loop: /new (new conversation), /quit.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import uuid

from ai_healthcare.schemas.conversation import ConversationState, TurnResult
from ai_healthcare.services.runtime import RuntimeConfig, build_runtime


def render(result: TurnResult) -> None:
    print(f"\n[{result.kind}] {result.latency_ms} ms")
    if result.proposed_plan is not None:
        proposed = [t.agent for t in result.proposed_plan.tasks]
        executed = (
            [t.agent for t in result.executed_plan.tasks]
            if result.executed_plan
            else None
        )
        print(f"  commander plan: {result.proposed_plan.intent} {proposed}")
        print(f"  executed plan : {executed}")
    if result.gate is not None:
        print(
            f"  gate          : allow={result.gate.allow} "
            f"reason={result.gate.reason_code} "
            f"limited={result.gate.limited_information}"
        )
    if result.symptom_output is not None:
        for m in result.symptom_output.symptoms:
            print(
                f"  symptom       : {m.name} ({m.polarity}) "
                f"onset={m.onset_text!r} duration={m.duration_text!r} "
                f"severity={m.severity_text!r}"
            )
    if result.questions:
        print("  questions:")
        for q in result.questions:
            print(f"    - {q}")
    if result.message:
        print(f"  message: {result.message}")

    out = result.medical_output
    if out is not None:
        print(f"  rag status    : {out.status}")
        for e in out.retrieved_evidence:
            print(
                f"  retrieved     : {e.evidence_id} {e.similarity_score:.3f} "
                f"{e.title} [{e.document_id}#{e.chunk_index}]"
            )
        print()
        if out.status == "answered":
            titles = {e.evidence_id: e.title for e in out.retrieved_evidence}
            for claim in out.claims:
                ids = ", ".join(
                    f"{i}={titles.get(i, '?')}" for i in claim.evidence_ids
                )
                print(f"  - {claim.text}\n      [{ids}]")
        else:
            print(out.answer_text)
        if result.language_outcome:
            print(f"  language check: {result.language_outcome}")
    if result.coverage_notice:
        print(f"\n  ({result.coverage_notice})")
    if result.language_notice:
        print(f"  ({result.language_notice})")


def main() -> int:
    default = RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=default.collection)
    args = parser.parse_args()

    config = dataclasses.replace(default, collection=args.collection)
    runtime = build_runtime(config)
    print(f"collection={config.collection} points={runtime.point_count}")
    print(f"rag prompt={config.rag_prompt_path.name}")

    state = ConversationState.new(uuid.uuid4().hex)
    try:
        while True:
            try:
                message = input("\nbạn> ")
            except (EOFError, KeyboardInterrupt):
                return 0
            command = message.strip()
            if command == "/quit":
                return 0
            if command == "/new":
                state = ConversationState.new(uuid.uuid4().hex)
                print("(đã bắt đầu hội thoại mới)")
                continue
            result = runtime.service.handle_turn(state, message)
            state = result.state
            render(result)
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
