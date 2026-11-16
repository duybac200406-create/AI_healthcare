
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_healthcare.evaluation.benchmarks.cadec_v2 import CadecV2Adapter


DATA_ROOT = Path(r"D:\datasets\CADECv2\extracted\raw\data")
RUN_DIR = Path(
    r"D:\AI Healthcare\reports\cadec_v2_dev_context_20261115T203858Z"
)

PREDICTIONS_PATH = RUN_DIR / "predictions.jsonl"


def main() -> int:
    adapter = CadecV2Adapter(DATA_ROOT)

    documents = adapter.load_split(
        "dev",
        include_discontinuous=True,
    )
    document_map = {
        doc.document_id: doc for doc in documents
    }

    with PREDICTIONS_PATH.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    category_counts = Counter()
    negative_category_counts = Counter()
    category_texts = defaultdict(Counter)
    category_examples = defaultdict(list)
    other_label_counts = Counter()

    structured_errors = []

    total_predictions = 0
    total_tp = 0
    negative_predictions = 0

    for row in rows:
        doc_id = row["document_id"]
        document = document_map[doc_id]
        source = document.text

        gold_counter = Counter(
            (span["start_char"], span["end_char"])
            for span in row["gold_spans"]
        )
        remaining_gold = gold_counter.copy()

        gold_spans = list(gold_counter)
        negative_document = len(row["gold_spans"]) == 0

        # All CADEC annotation surfaces and labels for this document.
        surface_labels = defaultdict(set)
        segment_labels = defaultdict(set)

        for mention in document.mentions:
            if mention.annotated_surface:
                surface_labels[mention.annotated_surface].add(
                    mention.label
                )

            for segment in mention.segments:
                segment_labels[
                    (segment.start, segment.end)
                ].add(mention.label)

        if row["status"] != "ok":
            structured_errors.append({
                "document_id": doc_id,
                "status": row["status"],
                "error": row.get("error"),
                "gold_count": len(row["gold_spans"]),
            })

        for prediction in row["predictions"]:
            total_predictions += 1
            text = prediction["text"]

            category = None
            labels = set()

            if not prediction["offset_valid"]:
                labels = surface_labels.get(text, set())

                if "Symptom" in labels:
                    category = "unresolved_symptom_gold_surface"
                elif labels:
                    category = "unresolved_other_label_surface"
                else:
                    category = "unresolved_no_gold_surface"

            else:
                start = prediction["start_char"]
                end = prediction["end_char"]
                key = (start, end)

                # Consume one exact Gold occurrence.
                if remaining_gold[key] > 0:
                    remaining_gold[key] -= 1
                    total_tp += 1
                    continue

                if key in segment_labels:
                    labels = segment_labels[key]

                    if "Symptom" in labels:
                        category = "matches_discontinuous_symptom_segment"
                    else:
                        category = "exact_other_label_span"

                elif text in surface_labels:
                    labels = surface_labels[text]

                    if "Symptom" in labels:
                        category = "symptom_surface_wrong_coordinates"
                    else:
                        category = "matches_other_label_surface"

                elif any(
                    max(start, gold_start) < min(end, gold_end)
                    for gold_start, gold_end in gold_spans
                ):
                    category = "overlaps_symptom_gold_span"

                else:
                    category = "no_exact_gold_surface_match"

            category_counts[category] += 1
            category_texts[category][text] += 1

            if labels:
                other_label_counts.update(labels)

            if negative_document:
                negative_predictions += 1
                negative_category_counts[category] += 1

            if len(category_examples[category]) < 5:
                if prediction["offset_valid"]:
                    context_start = max(0, start - 70)
                    context_end = min(len(source), end + 70)
                    context = source[context_start:context_end]
                else:
                    context = (
                        prediction.get("left_context", "")
                        + " [PREDICTION] "
                        + text
                        + " "
                        + prediction.get("right_context", "")
                    )

                category_examples[category].append({
                    "document_id": doc_id,
                    "text": text,
                    "labels_with_same_surface": sorted(labels),
                    "resolution_status": prediction["resolution_status"],
                    "context": context,
                })

    print("\n========== RUN VALIDATION ==========")
    print("Documents:", len(rows))
    print("Predictions:", total_predictions)
    print("Recomputed TP:", total_tp)
    print("Negative-document predictions:", negative_predictions)

    print("\n========== FALSE POSITIVE CATEGORIES ==========")
    for category, count in category_counts.most_common():
        print(f"\n{category}: {count}")

        print("Common predicted texts:")
        for text, frequency in category_texts[category].most_common(10):
            print(f"  {frequency:>3} | {text!r}")

        print("Examples:")
        for example in category_examples[category]:
            print(example)

    print("\n========== FALSE POSITIVES IN NEGATIVE DOCUMENTS ==========")
    for category, count in negative_category_counts.most_common():
        print(category, count)

    print("\n========== MATCHING ANNOTATION LABELS ==========")
    print(dict(other_label_counts))

    print("\n========== STRUCTURED OUTPUT ERRORS ==========")
    for error in structured_errors:
        print(error)

    print("\nError analysis complete.")
    print("No source files were modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
