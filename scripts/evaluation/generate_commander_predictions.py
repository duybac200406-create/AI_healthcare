
"""Generate real Commander routing predictions using local Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_healthcare.agents.commander import Commander
from ai_healthcare.providers.ollama_provider import OllamaProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/commander_routing_v0.1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/commander_predictions_v0.1.jsonl"),
    )
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig") as file:
        cases = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with OllamaProvider(
        model="qwen3:8b",
        temperature=0,
        num_ctx=4096,
        num_predict=512,
        timeout=120,
        no_think=True,
    ) as provider:
        commander = Commander(provider)

        with args.output.open("w", encoding="utf-8") as output_file:
            for index, case in enumerate(cases, start=1):
                case_id = case["case_id"]
                user_message = case["user_message"]

                plan = commander.plan(user_message)

                row = {
                    "case_id": case_id,
                    "output": plan.model_dump(mode="json"),
                }

                output_file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                print(
                    f"[{index}/{len(cases)}] {case_id}: "
                    f"{plan.status} | {plan.intent} | "
                    f"{[task.agent for task in plan.tasks]}"
                )

    print(f"\nPredictions saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
