import pytest

from ai_healthcare.schemas.symptom import SymptomExtractionOutput
from scripts.evaluate_symptom_agent import (
    score_case,
    validate_case_annotations,
)


def make_case():
    return {
        "id": "TEST01",
        "message": "Em ăn gì cũng thấy khó chịu.",
        "expected_status": "extracted",
        "expected_symptoms": [
            {
                "expected_name": "khó chịu khi ăn",
                "aliases": ["khó chịu khi ăn"],
                "polarity": "present",
                "expected_reported_text": "ăn gì cũng thấy khó chịu",
            }
        ],
    }

def make_output(reported_text):
    return SymptomExtractionOutput(
        status="extracted",
        symptoms=(
            {
                "name": "khó chịu khi ăn",
                "reported_text": reported_text,
                "polarity": "present",
                "onset_text": None,
                "duration_text": None,
                "severity_text": None,
                "body_site_text": None,
            },
        ),
    )
def make_output_from_mentions(mentions):
    return SymptomExtractionOutput(
        status="extracted",
        symptoms=tuple(
            {
                "name": item["name"],
                "reported_text": item["reported_text"],
                "polarity": item["polarity"],
                "onset_text": item.get("onset_text"),
                "duration_text": item.get("duration_text"),
                "severity_text": item.get("severity_text"),
                "body_site_text": item.get("body_site_text"),
            }
            for item in mentions
        ),
    )
def make_raw_payload(reported_text):
    return {
        "symptoms": [
            {
                "name": "khó chịu khi ăn",
                "reported_text": reported_text,
                "polarity": "present",
            }
        ]
    }
def test_raw_and_resolved_reported_text_are_scored_separately():
    case = make_case()

    # Raw Qwen output contains an incomplete span.
    raw_payload = make_raw_payload("thấy khó chịu")

    # Span Resolver has corrected the span.
    output = make_output("ăn gì cũng thấy khó chịu")

    result = score_case(case, output, raw_payload)

    assert result["raw_reported_text_total"] == 1
    assert result["raw_reported_text_correct"] == 0
    assert result["raw_reported_text_accuracy"] == 0.0

    assert result["reported_text_total"] == 1
    assert result["reported_text_correct"] == 1
    assert result["reported_text_accuracy"] == 1.0

    check = result["reported_text_checks"][0]

    assert check["raw_predicted"] == "thấy khó chịu"
    assert check["raw_correct"] is False

    assert check["predicted"] == "ăn gì cũng thấy khó chịu"
    assert check["correct"] is True
def test_raw_final_alignment_rejects_different_symptom_counts():
    case = make_case()
    output = make_output("ăn gì cũng thấy khó chịu")

    raw_payload = {
        "symptoms": []
    }

    with pytest.raises(
        ValueError,
        match="Raw/final symptom count mismatch",
    ):
        score_case(case, output, raw_payload)
def test_raw_final_alignment_rejects_reordered_symptoms():
    case = make_case()

    output = SymptomExtractionOutput(
        status="extracted",
        symptoms=(
            {
                "name": "sốt",
                "reported_text": "sốt",
                "polarity": "present",
                "onset_text": None,
                "duration_text": None,
                "severity_text": None,
                "body_site_text": None,
            },
            {
                "name": "ho",
                "reported_text": "ho",
                "polarity": "present",
                "onset_text": None,
                "duration_text": None,
                "severity_text": None,
                "body_site_text": None,
            },
        ),
    )

    raw_payload = {
        "symptoms": [
            {
                "name": "ho",
                "reported_text": "ho",
                "polarity": "present",
            },
            {
                "name": "sốt",
                "reported_text": "sốt",
                "polarity": "present",
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="Raw/final name mismatch",
    ):
        score_case(case, output, raw_payload)        
def test_reported_text_requires_exact_expected_span():
    case = make_case()

    # This is a valid substring but not the expected complete span.
    output = make_output("thấy khó chịu")

    result = score_case(case, output)

    assert result["reported_text_total"] == 1
    assert result["reported_text_correct"] == 0
    assert result["reported_text_accuracy"] == 0.0


def test_reported_text_accepts_exact_expected_span():
    case = make_case()
    output = make_output("ăn gì cũng thấy khó chịu")

    result = score_case(case, output)

    assert result["reported_text_total"] == 1
    assert result["reported_text_correct"] == 1
    assert result["reported_text_accuracy"] == 1.0


def test_case_annotation_must_be_exact_substring():
    case = make_case()

    validate_case_annotations([case])

    case["expected_symptoms"][0]["expected_reported_text"] = (
        "khó chịu khi ăn"
    )

    with pytest.raises(ValueError, match="exact substring"):
        validate_case_annotations([case])
def test_span_accuracy_is_independent_of_normalized_name_accuracy():
    case = {
        "id": "TEST_SPAN_01",
        "message": "Tôi thấy đầu quay cuồng.",
        "expected_status": "extracted",
        "expected_symptoms": [
            {
                "expected_name": "chóng mặt",
                "aliases": ["chóng mặt", "choáng váng"],
                "polarity": "present",
                "expected_reported_text": "đầu quay cuồng",
            }
        ],
    }

    raw_symptoms = [
        {
            "name": "đau đầu",
            "reported_text": "đầu quay cuồng",
            "polarity": "present",
        }
    ]

    output = make_output_from_mentions(raw_symptoms)

    raw_payload = {
        "symptoms": raw_symptoms,
    }

    result = score_case(case, output, raw_payload)

    # Name is wrong.
    assert result["normalized_name_total"] == 1
    assert result["normalized_name_correct"] == 0
    assert result["normalized_name_accuracy"] == 0.0

    # Source spans are correct independently of the name.
    assert result["raw_reported_text_total"] == 1
    assert result["raw_reported_text_correct"] == 1
    assert result["raw_reported_text_accuracy"] == 1.0

    assert result["reported_text_total"] == 1
    assert result["reported_text_correct"] == 1
    assert result["reported_text_accuracy"] == 1.0

    # A wrong symptom label must not become a true positive.
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
def test_candidate_alignment_prioritizes_exact_source_span_before_polarity():
    case = {
        "id": "TEST_SPAN_02",
        "message": "Tôi bị đau đầu.",
        "expected_status": "extracted",
        "expected_symptoms": [
            {
                "expected_name": "đau đầu",
                "aliases": ["đau đầu", "nhức đầu"],
                "polarity": "present",
                "expected_reported_text": "đau đầu",
            }
        ],
    }

    raw_symptoms = [
        {
            # Exact source span, but wrong polarity.
            "name": "đau đầu",
            "reported_text": "đau đầu",
            "polarity": "uncertain",
        },
        {
            # Correct polarity, but longer source span.
            "name": "đau đầu",
            "reported_text": "bị đau đầu",
            "polarity": "present",
        },
    ]

    output = make_output_from_mentions(raw_symptoms)

    raw_payload = {
        "symptoms": raw_symptoms,
    }

    result = score_case(case, output, raw_payload)

    # The exact source span must be selected.
    check = result["reported_text_checks"][0]

    assert check["raw_predicted"] == "đau đầu"
    assert check["predicted"] == "đau đầu"

    # Name must be counted exactly once.
    assert result["normalized_name_total"] == 1
    assert result["normalized_name_correct"] == 1
    assert result["normalized_name_accuracy"] == 1.0

    # Span is correct, but the selected prediction has wrong polarity.
    assert result["raw_reported_text_correct"] == 1
    assert result["reported_text_correct"] == 1

    assert result["correct_polarities"] == 0
    assert result["true_positives"] == 0
def test_attribute_exact_match_accepts_correct_value():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "dữ dội",
        }
    ]

    output = make_output_from_mentions([
        {
            "name": "khó chịu khi ăn",
            "reported_text": "ăn gì cũng thấy khó chịu",
            "polarity": "present",
            "severity_text": "dữ dội",
        }
    ])

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 1
    assert result["attribute_exact_match_accuracy"] == 1.0

    assert result["attribute_accuracy_by_field"]["severity_text"] == {
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
def test_attribute_exact_match_rejects_wrong_value():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "dữ dội",
        }
    ]

    output = make_output_from_mentions([
        {
            "name": "khó chịu khi ăn",
            "reported_text": "ăn gì cũng thấy khó chịu",
            "polarity": "present",
            "severity_text": "liên tục",
        }
    ])

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 0
    assert result["attribute_exact_match_accuracy"] == 0.0

    check = result["attribute_checks"][0]

    assert check["expected"] == "dữ dội"
    assert check["predicted"] == "liên tục"
    assert check["correct"] is False
def test_missing_expected_attribute_counts_as_incorrect():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "duration_text",
            "expected_value": "ba ngày nay",
        }
    ]

    output = make_output(
        "ăn gì cũng thấy khó chịu"
    )

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 0
    assert result["attribute_exact_match_accuracy"] == 0.0

    check = result["attribute_checks"][0]

    assert check["predicted"] is None
    assert check["correct"] is False
def test_attribute_annotation_must_be_exact_substring():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "mức độ nghiêm trọng",
        }
    ]

    with pytest.raises(
        ValueError,
        match="not an exact substring",
    ):
        validate_case_annotations([case])
def test_attribute_annotation_rejects_unknown_field():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "diagnosis",
            "expected_value": "viêm dạ dày",
        }
    ]

    with pytest.raises(
        ValueError,
        match="unsupported attribute field",
    ):
        validate_case_annotations([case])
def test_attribute_exact_match_accepts_correct_value():
    case = make_case()
    case["message"] = "Em ăn gì cũng thấy khó chịu dữ dội."

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "dữ dội",
        }
    ]

    output = make_output_from_mentions([
        {
            "name": "khó chịu khi ăn",
            "reported_text": "ăn gì cũng thấy khó chịu",
            "polarity": "present",
            "severity_text": "dữ dội",
        }
    ])

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 1
    assert result["attribute_exact_match_accuracy"] == 1.0
def test_attribute_exact_match_rejects_wrong_value():
    case = make_case()
    case["message"] = "Em ăn gì cũng thấy khó chịu dữ dội."

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "dữ dội",
        }
    ]

    output = make_output_from_mentions([
        {
            "name": "khó chịu khi ăn",
            "reported_text": "ăn gì cũng thấy khó chịu",
            "polarity": "present",
            "severity_text": "liên tục",
        }
    ])

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 0
    assert result["attribute_exact_match_accuracy"] == 0.0

    check = result["attribute_checks"][0]
    assert check["expected"] == "dữ dội"
    assert check["predicted"] == "liên tục"
    assert check["correct"] is False
def test_missing_expected_attribute_counts_as_incorrect():
    case = make_case()
    case["message"] = (
        "Em ăn gì cũng thấy khó chịu suốt ba ngày nay."
    )

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "duration_text",
            "expected_value": "ba ngày nay",
        }
    ]

    output = make_output(
        "ăn gì cũng thấy khó chịu"
    )

    result = score_case(case, output)

    assert result["attribute_exact_match_total"] == 1
    assert result["attribute_exact_match_correct"] == 0
    assert result["attribute_exact_match_accuracy"] == 0.0

    check = result["attribute_checks"][0]
    assert check["predicted"] is None
    assert check["correct"] is False
def test_attribute_annotation_must_be_exact_substring():
    case = make_case()

    case["expected_symptoms"][0]["expected_fields"] = [
        {
            "field": "severity_text",
            "expected_value": "mức độ nghiêm trọng",
        }
    ]

    with pytest.raises(
        ValueError,
        match="not an exact substring",
    ):
        validate_case_annotations([case])