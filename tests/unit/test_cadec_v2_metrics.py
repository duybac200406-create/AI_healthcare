
from scripts.evaluate_cadec_v2 import aggregate_document_metrics


def test_offsets_from_different_documents_must_not_match():
    rows = [
        {
            "document_id": "DOC-A",
            "gold_spans": [
                {
                    "start_char": 10,
                    "end_char": 15,
                    "text": "pain",
                }
            ],
            "predictions": [],
        },
        {
            "document_id": "DOC-B",
            "gold_spans": [],
            "predictions": [
                {
                    "text": "other",
                    "start_char": 10,
                    "end_char": 15,
                    "offset_valid": True,
                }
            ],
        },
    ]

    result = aggregate_document_metrics(rows)

    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["gold"] == 1
    assert result["predicted"] == 1
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_duplicate_predictions_are_still_false_positives():
    prediction = {
        "text": "headache",
        "start_char": 20,
        "end_char": 28,
        "offset_valid": True,
    }

    rows = [
        {
            "document_id": "DOC-C",
            "gold_spans": [
                {
                    "start_char": 20,
                    "end_char": 28,
                    "text": "headache",
                }
            ],
            "predictions": [prediction, prediction],
        }
    ]

    result = aggregate_document_metrics(rows)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
