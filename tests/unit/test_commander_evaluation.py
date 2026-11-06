"""Unit tests for the Commander routing evaluator."""

from pathlib import Path

import pytest

from ai_healthcare.evaluation.commander import evaluate_records, load_jsonl

DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "commander_routing_v0.1.jsonl"
)


def _case() -> dict:
    return {
        "case_id": "CMD-T001",
        "category": "medical_information",
        "user_message": "Dị ứng latex là gì?",
        "expected_status": "ready",
        "expected_intent": "medical_information",
        "expected_agents": ["medical_rag"],
        "rationale": "General information.",
    }


def _prediction(output: dict | str, case_id: str = "CMD-T001") -> dict:
    return {"case_id": case_id, "output": output}


def _valid_output() -> dict:
    return {
        "intent": "medical_information",
        "status": "ready",
        "tasks": [
            {
                "task_id": "task_1",
                "agent": "medical_rag",
                "objective": "Retrieve approved general evidence about latex allergy.",
                "depends_on": [],
            }
        ],
        "clarification_question": None,
    }


def test_dataset_contains_unique_cases_and_patient_record_boundary() -> None:
    cases = load_jsonl(DATASET)
    case_ids = [case["case_id"] for case in cases]

    assert len(cases) == 16
    assert len(case_ids) == len(set(case_ids))
    patient_record_case = next(
        case for case in cases if case["case_id"] == "CMD-016"
    )
    assert patient_record_case["expected_status"] == "unsupported"
    assert patient_record_case["expected_agents"] == []


def test_exact_route_match_is_scored_correctly() -> None:
    report = evaluate_records([_case()], [_prediction(_valid_output())])

    assert report["valid_plan_count"] == 1
    assert report["valid_plan_rate"] == 1.0
    assert report["exact_route_match_rate"] == 1.0


def test_invalid_agent_plan_is_not_counted_as_valid() -> None:
    output = _valid_output()
    output["tasks"][0]["agent"] = "patient_record"

    report = evaluate_records([_case()], [_prediction(output)])

    assert report["valid_plan_count"] == 0
    assert report["exact_route_match_rate"] == 0.0
    assert report["cases"][0]["valid_plan"] is False


def test_missing_prediction_counts_against_full_dataset() -> None:
    report = evaluate_records([_case()], [])

    assert report["valid_plan_count"] == 0
    assert report["valid_plan_rate"] == 0.0
    assert report["cases"][0]["errors"] == ["missing_prediction"]


def test_duplicate_prediction_ids_are_not_silently_accepted() -> None:
    predictions = [
        _prediction(_valid_output()),
        _prediction(_valid_output()),
    ]

    report = evaluate_records([_case()], predictions)

    assert report["duplicate_prediction_ids"] == ["CMD-T001"]
    assert report["valid_plan_count"] == 0


def test_json_string_output_is_accepted() -> None:
    import json

    report = evaluate_records(
        [_case()], [_prediction(json.dumps(_valid_output()))]
    )

    assert report["valid_plan_count"] == 1
    assert report["exact_route_match_rate"] == 1.0


def test_duplicate_case_ids_in_dataset_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate case_id"):
        evaluate_records([_case(), _case()], [])
