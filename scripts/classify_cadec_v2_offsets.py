
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_healthcare.evaluation.benchmarks.cadec_v2 import CadecV2Adapter


DATA_ROOT = Path(r"D:\datasets\CADECv2\extracted\raw\data")

TARGET_SPLITS = ("train", "dev")
AUDIT_LABELS = {"ADR", "Symptom"}
TARGET_KIND = "discontinuous"

EXPECTED_COUNTS = {
    "train": {"ADR": 1805, "Symptom": 64},
    "dev": {"ADR": 223, "Symptom": 6},
}



def classify_segments(
    raw_text: str,
    lf_text: str,
    segments: list[tuple[int, int]],
    surface: str,
) -> tuple[str, dict]:
    """
    Audit discontinuous spans without assuming that annotation-order
    segments must be in document order.
    """

    if len(segments) < 2:
        return "invalid_segments", {
            "reason": "fewer_than_two_segments"
        }

    # Validate each individual segment.
    for start, end in segments:
        if start < 0 or end <= start:
            return "invalid_segments", {
                "reason": "invalid_start_end",
                "segments": segments,
            }

    # Check overlap independently of the order in the .ann file.
    document_segments = sorted(segments, key=lambda item: (item[0], item[1]))

    has_overlap = any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(
            document_segments,
            document_segments[1:],
        )
    )

    if has_overlap:
        return "overlapping_segments", {
            "segments": segments,
            "document_segments": document_segments,
        }

    document_ordered = segments == document_segments

    raw_valid = all(
        0 <= start < end <= len(raw_text)
        for start, end in segments
    )

    lf_valid = all(
        0 <= start < end <= len(lf_text)
        for start, end in segments
    )

    if not raw_valid and not lf_valid:
        return "out_of_bounds", {
            "segments": segments,
            "raw_valid": raw_valid,
            "lf_valid": lf_valid,
            "document_ordered": document_ordered,
        }

    def join_parts(
        text: str,
        ordered_segments: list[tuple[int, int]],
        valid: bool,
    ) -> str | None:
        if not valid:
            return None

        return " ".join(
            text[start:end]
            for start, end in ordered_segments
        )

    # Compare using the exact segment order stored in .ann.
    raw_annotation_joined = join_parts(
        raw_text, segments, raw_valid
    )
    lf_annotation_joined = join_parts(
        lf_text, segments, lf_valid
    )

    # Also compare using ascending document offsets.
    raw_document_joined = join_parts(
        raw_text, document_segments, raw_valid
    )
    lf_document_joined = join_parts(
        lf_text, document_segments, lf_valid
    )

    raw_annotation_match = raw_annotation_joined == surface
    lf_annotation_match = lf_annotation_joined == surface

    raw_document_match = raw_document_joined == surface
    lf_document_match = lf_document_joined == surface

    raw_match = raw_annotation_match
    lf_match = lf_annotation_match

    if raw_match and lf_match:
        result = "both_match"
    elif raw_match:
        result = "raw_only"
    elif lf_match:
        result = "lf_only"
    else:
        result = "neither"

    details = {
        "segments": segments,
        "document_segments": document_segments,
        "document_ordered": document_ordered,
        "raw_valid": raw_valid,
        "lf_valid": lf_valid,
        "raw_annotation_joined": raw_annotation_joined,
        "lf_annotation_joined": lf_annotation_joined,
        "raw_document_joined": raw_document_joined,
        "lf_document_joined": lf_document_joined,
        "raw_annotation_match": raw_annotation_match,
        "lf_annotation_match": lf_annotation_match,
        "raw_document_match": raw_document_match,
        "lf_document_match": lf_document_match,
    }

    return result, details


def main() -> int:
    adapter = CadecV2Adapter(DATA_ROOT)

    validation = adapter.validate_splits()
    if not validation.get("valid"):
        raise RuntimeError("Official split validation failed.")

    for split in TARGET_SPLITS:
        documents = adapter.load_split(
            split,
            labels=AUDIT_LABELS,
            include_discontinuous=True,
        )

        stats = defaultdict(Counter)
        order_stats = defaultdict(Counter)
        anomaly_examples = defaultdict(
            lambda: defaultdict(list)
        )

        observed_counts = Counter()

        for document in documents:
            doc_id = document.document_id

            txt_files = list(
                (DATA_ROOT / "txt").rglob(doc_id + ".txt")
            )
            ann_files = list(
                (DATA_ROOT / "ann").rglob(doc_id + ".ann")
            )

            if len(txt_files) != 1 or len(ann_files) != 1:
                raise RuntimeError(
                    f"Missing or duplicate files: {doc_id}"
                )

            raw_text = txt_files[0].read_bytes().decode("utf-8")

            lf_text = raw_text.replace(
                "\r\n", "\n"
            ).replace("\r", "\n")

            lines = ann_files[0].read_text(
                encoding="utf-8"
            ).splitlines()

            for line_no, line in enumerate(lines, start=1):
                fields = line.split("\t", 2)

                if len(fields) != 3 or not fields[0].startswith("T"):
                    continue

                description = fields[1].split(maxsplit=1)

                if len(description) != 2:
                    continue

                label, expression = description

                if label not in AUDIT_LABELS:
                    continue

                try:
                    segments = []

                    for part in expression.split(";"):
                        values = part.split()

                        if len(values) != 2:
                            raise ValueError(
                                f"Invalid segment expression: {part}"
                            )

                        start, end = map(int, values)
                        segments.append((start, end))

                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid annotation: {doc_id}, line {line_no}"
                    ) from exc

                if len(segments) < 2:
                    continue

                observed_counts[label] += 1

                surface = fields[2]

                result, details = classify_segments(
                    raw_text,
                    lf_text,
                    segments,
                    surface,
                )

                stats[label][result] += 1
                if details.get("document_ordered"):
                    order_stats[label]["document_ordered"] += 1
                else:
                    order_stats[label]["out_of_document_order"] += 1
                if result != "both_match" or not details.get("document_ordered", True):
                    category = (
                        result
                        if result != "both_match"
                        else "out_of_document_order"
                    )
                
                    examples = anomaly_examples[label][category]
                
                    if len(examples) < 10:
                        examples.append({
                            "document_id": doc_id,
                            "line_no": line_no,
                            "entity_id": fields[0],
                            "label": label,
                            "surface": surface,
                            **details,
                        })

        print(f"\n========== {split.upper()} ==========")

        split_total = Counter()

        for label in sorted(AUDIT_LABELS):
            print(f"\n--- {label} ---")
            print(
                "Segment order:",
                dict(order_stats[label]),
            )
            print(
                f"{TARGET_KIND}:",
                dict(stats[label]),
            )

            print(
                "Observed mentions:",
                observed_counts[label],
            )

            expected = EXPECTED_COUNTS[split][label]

            print(
                f"Expected mentions: {expected}"
            )

            print(
                "Count check:",
                "PASS" if observed_counts[label] == expected else "FAIL",
            )

            split_total.update(stats[label])

        print("\nSPLIT TOTAL:", dict(split_total))

        for label in sorted(AUDIT_LABELS):
            for category, examples in anomaly_examples[label].items():
                print(f"\n=== {label} / {category.upper()} ===")

                for example in examples:
                    print(example)

    print("\nDiscontinuous offset classification complete.")
    print("No source files were modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
