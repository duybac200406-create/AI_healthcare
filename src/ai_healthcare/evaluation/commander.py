"""Evaluate Commander routing predictions against a small labelled JSONL set.

Prediction JSONL format (one row per case):
{"case_id": "CMD-001", "output": { ... CommanderPlan ... }}

`output` may also be a JSON string. This evaluator scores routing only; it does
not score medical correctness, the quality of task objectives, or clinical safety.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.schemas.agent import CommanderPlan


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines and require every line to contain a JSON object."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            rows.append(value)
    return rows


def _normalise_output(value: Any) -> dict[str, Any]:
    """Accept either a decoded JSON object or a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Prediction 'output' must be a JSON object or JSON string.")
    return value


def evaluate_records(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate predictions and calculate routing metrics.

    Exact route match requires status, intent, and the set of selected agents to
    match the labelled expectation. Task wording and clarification wording are
    deliberately not scored automatically and require human review.
    """
    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every evaluation case must have a non-empty case_id.")
        if case_id in case_by_id:
            raise ValueError(f"Duplicate case_id in dataset: {case_id}")
        for field in ("expected_status", "expected_intent", "expected_agents"):
            if field not in case:
                raise ValueError(f"Case {case_id} is missing '{field}'.")
        if not isinstance(case["expected_agents"], list):
            raise ValueError(f"Case {case_id}: expected_agents must be a list.")
        case_by_id[case_id] = case

    prediction_counts = Counter(
        row.get("case_id") for row in predictions if isinstance(row.get("case_id"), str)
    )
    predictions_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id not in predictions_by_id:
            predictions_by_id[case_id] = row

    orchestrator = Orchestrator()
    case_results: list[dict[str, Any]] = []
    valid_count = 0
    status_correct = 0
    intent_correct = 0
    agent_set_correct = 0
    exact_route_correct = 0

    for case_id, case in case_by_id.items():
        errors: list[str] = []
        prediction = predictions_by_id.get(case_id)
        plan: CommanderPlan | None = None

        if prediction is None:
            errors.append("missing_prediction")
        elif prediction_counts[case_id] > 1:
            errors.append("duplicate_prediction_case_id")
        else:
            try:
                raw_output = _normalise_output(prediction.get("output"))
                plan = orchestrator.validate_plan(raw_output)
                valid_count += 1
            except (ValueError, TypeError, ValidationError) as exc:
                errors.append(f"invalid_plan: {exc}")

        expected_agents = sorted(case["expected_agents"])
        predicted_agents = sorted({task.agent for task in plan.tasks}) if plan else None
        status_match = plan is not None and plan.status == case["expected_status"]
        intent_match = plan is not None and plan.intent == case["expected_intent"]
        agents_match = plan is not None and predicted_agents == expected_agents
        exact_match = bool(status_match and intent_match and agents_match)

        status_correct += int(status_match)
        intent_correct += int(intent_match)
        agent_set_correct += int(agents_match)
        exact_route_correct += int(exact_match)

        case_results.append(
            {
                "case_id": case_id,
                "category": case.get("category"),
                "valid_plan": plan is not None,
                "expected": {
                    "status": case["expected_status"],
                    "intent": case["expected_intent"],
                    "agents": expected_agents,
                },
                "predicted": (
                    {
                        "status": plan.status,
                        "intent": plan.intent,
                        "agents": predicted_agents,
                        "task_count": len(plan.tasks),
                        "clarification_question_present": bool(
                            plan.clarification_question
                        ),
                    }
                    if plan
                    else None
                ),
                "status_match": status_match,
                "intent_match": intent_match,
                "agent_set_match": agents_match,
                "exact_route_match": exact_match,
                "errors": errors,
            }
        )

    unknown_prediction_ids = sorted(set(predictions_by_id) - set(case_by_id))
    total = len(case_by_id)
    denominator = total if total else 1

    return {
        "evaluation_name": "commander_routing_v0.1",
        "scope": "routing/schema validity only; not medical correctness or clinical safety",
        "total_cases": total,
        "prediction_rows": len(predictions),
        "valid_plan_count": valid_count,
        "valid_plan_rate": valid_count / denominator,
        "status_accuracy": status_correct / denominator,
        "intent_accuracy": intent_correct / denominator,
        "agent_set_accuracy": agent_set_correct / denominator,
        "exact_route_match_rate": exact_route_correct / denominator,
        "unknown_prediction_ids": unknown_prediction_ids,
        "duplicate_prediction_ids": sorted(
            case_id for case_id, count in prediction_counts.items() if count > 1
        ),
        "cases": case_results,
        "interpretation_warning": (
            "Small manually authored pilot set. Results are diagnostic, not a "
            "validated benchmark and not evidence of clinical safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/commander_routing_v0.1.jsonl"),
        help="Labelled routing dataset (JSONL).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Model predictions in JSONL format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the evaluation report as JSON.",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    predictions = load_jsonl(args.predictions)
    report = evaluate_records(cases, predictions)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
