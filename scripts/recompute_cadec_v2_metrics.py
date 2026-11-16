
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cadec_v2 import aggregate_document_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute CADECv2 metrics offline from saved predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the existing predictions.jsonl file.",
    )
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=354,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to corrected_summary.json beside predictions.",
    )
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    rows: list[dict[str, Any]] = []
    document_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")

            document_id = row.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(
                    f"Missing document_id at line {line_number}."
                )

            if document_id in document_ids:
                raise ValueError(
                    f"Duplicate document_id found: {document_id}"
                )

            document_ids.add(document_id)

            if not isinstance(row.get("gold_spans"), list):
                raise ValueError(
                    f"Invalid gold_spans in document {document_id}."
                )

            if not isinstance(row.get("predictions"), list):
                raise ValueError(
                    f"Invalid predictions in document {document_id}."
                )

            for prediction in row["predictions"]:
                if not isinstance(prediction.get("offset_valid"), bool):
                    raise ValueError(
                        f"Invalid offset_valid in document {document_id}."
                    )

                if prediction["offset_valid"]:
                    start = prediction.get("start_char")
                    end = prediction.get("end_char")

                    if (
                        not isinstance(start, int)
                        or isinstance(start, bool)
                        or not isinstance(end, int)
                        or isinstance(end, bool)
                        or start < 0
                        or end <= start
                    ):
                        raise ValueError(
                            f"Invalid resolved offsets in {document_id}."
                        )

            rows.append(row)

    if not rows:
        raise ValueError("Predictions file contains no records.")

    return rows


def main() -> int:
    args = parse_args()

    if args.expected_documents <= 0:
        print("[FAIL] expected-documents must be positive.")
        return 1

    try:
        rows = load_prediction_rows(args.predictions)

        totals = aggregate_document_metrics(rows)

        status_counts = Counter(
            row.get("status", "unknown") for row in rows
        )

        total_gold = sum(len(row["gold_spans"]) for row in rows)
        total_predictions = sum(len(row["predictions"]) for row in rows)

        if totals["gold"] != total_gold:
            raise ValueError("Gold count mismatch after aggregation.")

        if totals["predicted"] != total_predictions:
            raise ValueError("Prediction count mismatch after aggregation.")

        output_path = args.output or (
            args.predictions.parent / "corrected_summary.json"
        )

        summary = {
            "benchmark": "CADECv2",
            "track": "continuous Symptom span extraction",
            "split": "dev",
            "evaluation_method": "document_scoped_strict_span_matching",
            "inference_performed": False,
            "source_predictions": str(args.predictions),
            "documents_processed": len(rows),
            "expected_documents": args.expected_documents,
            "full_dev_coverage": len(rows) == args.expected_documents,
            "gold_spans": total_gold,
            "predicted_spans": total_predictions,
            "status_counts": dict(status_counts),
            "model_output_error_documents": status_counts.get(
                "model_output_error", 0
            ),
            "infrastructure_error_documents": status_counts.get(
                "infrastructure_error", 0
            ),
            "metrics": totals,
            "notes": [
                "Metrics are recomputed from saved predictions; no model inference was performed.",
                "Gold and predictions are matched independently within each document.",
                "Unresolved and duplicate predictions remain false positives.",
                "The original summary.json is not overwritten.",
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("========== CADECv2 OFFLINE METRIC RECOMPUTATION ==========")
        print(f"Documents: {len(rows)}/{args.expected_documents}")
        print(f"Gold spans: {total_gold}")
        print(f"Predicted spans: {total_predictions}")
        print(f"TP={totals['tp']} FP={totals['fp']} FN={totals['fn']}")
        print(
            f"Precision={totals['precision']:.4f} "
            f"Recall={totals['recall']:.4f} "
            f"F1={totals['f1']:.4f}"
        )
        print(f"Output: {output_path}")

        if len(rows) != args.expected_documents:
            print("[WARN] Document count differs from expected Dev coverage.")

        return 0

    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
