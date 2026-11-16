
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_healthcare.evaluation.benchmarks.cadec_v2 import CadecV2Adapter


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "CADEC_V2_DATA",
        r"D:\datasets\CADECv2\extracted\raw\data",
    )
)

EXPECTED_SPLITS = {
    "train": 2485,
    "dev": 354,
    "test": 709,
}

# Canonical labels are used for reporting.
# Raw labels are always preserved separately.
def canonical_label(raw: str) -> str:
    key = re.sub(r"[^a-z]", "", raw.casefold())

    aliases = {
        "adr": "ADR",
        "ade": "ADR",
        "adversedrugevent": "ADR",
        "adversedrugreaction": "ADR",
        "symptom": "Symptom",
        "disease": "Disease",
        "drug": "Drug",
        "finding": "Finding",
    }

    return aliases.get(key, "UNKNOWN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline audit of original CADECv2 BRAT annotations. "
            "Train and Dev annotation contents only; Test remains untouched."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports",
    )

    return parser.parse_args()


def index_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    result: dict[str, Path] = {}

    for path in directory.rglob(f"*{suffix}"):
        if not path.is_file():
            continue

        key = path.stem

        if key in result:
            raise ValueError(
                f"Duplicate file ID {key} in {directory}: "
                f"{result[key]} and {path}"
            )

        result[key] = path

    return result


def parse_span_expression(expression: str) -> list[tuple[int, int]]:
    segments = []

    for part in expression.split(";"):
        values = part.strip().split()

        if len(values) != 2:
            raise ValueError(f"Invalid span segment: {part!r}")

        start, end = map(int, values)

        if start < 0 or end <= start:
            raise ValueError(f"Invalid offsets: {start}, {end}")

        segments.append((start, end))

    return segments


def parse_ann_file(
    ann_path: Path,
    source_text: str,
) -> dict[str, Any]:
    entities = []
    entity_ids: set[str] = set()

    record_counts = Counter()
    attribute_types = Counter()
    relation_types = Counter()
    normalization_records = []
    errors = []
    offset_checks = Counter()
    surface_mismatch_examples = []
    unparsed_records = []
    raw_labels = Counter()

    lines = ann_path.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        fields = line.split("\t", 2)
        record_id = fields[0]
        record_type = record_id[:1]

        record_counts[record_type] += 1

        try:
            if record_type == "T":
                if len(fields) < 3:
                    raise ValueError("Entity record has fewer than 3 fields")

                description = fields[1].split(maxsplit=1)

                if len(description) != 2:
                    raise ValueError("Invalid entity label/span description")

                raw_label, span_expression = description
                segments = parse_span_expression(span_expression)
                surface = fields[2]

                if record_id in entity_ids:
                    raise ValueError(f"Duplicate entity ID: {record_id}")

                entity_ids.add(record_id)
                raw_labels[raw_label] += 1

                valid_segments = True

                for start, end in segments:
                    if end > len(source_text):
                        valid_segments = False
                        offset_checks["out_of_bounds"] += 1
                    else:
                        offset_checks["segments_checked"] += 1

                if valid_segments:
                    extracted_parts = [
                        source_text[start:end]
                        for start, end in segments
                    ]

                    if len(segments) == 1:
                        surface_matches = extracted_parts[0] == surface
                    else:
                        surface_matches = (
                            " ".join(extracted_parts) == surface
                        )

                    if surface_matches:
                        offset_checks["surface_matches"] += 1
                    else:
                        offset_checks["surface_mismatches"] += 1

                        if len(surface_mismatch_examples) < 100:
                            surface_mismatch_examples.append({
                                "entity_id": record_id,
                                "raw_label": raw_label,
                                "surface": surface,
                                "extracted_parts": extracted_parts,
                                "segments": segments,
                                "discontinuous": len(segments) > 1,
                            })
                else:
                    surface_matches = False

                entities.append({
                    "entity_id": record_id,
                    "raw_label": raw_label,
                    "canonical_label": canonical_label(raw_label),
                    "segments": segments,
                    "discontinuous": len(segments) > 1,
                    "surface": surface,
                    "offset_valid": valid_segments,
                    "surface_matches": surface_matches,
                })

            elif record_type in {"A", "M"}:
                if len(fields) >= 2:
                    attribute_type = fields[1].split()[0]
                    attribute_types[attribute_type] += 1

            elif record_type == "R":
                if len(fields) >= 2:
                    relation_type = fields[1].split()[0]
                    relation_types[relation_type] += 1

            elif record_type == "N":
                if len(fields) >= 2:
                    tokens = fields[1].split()

                    if len(tokens) >= 3 and tokens[0] == "Reference":
                        entity_id = tokens[1]
                        ontology = tokens[2].split(":")[0]

                        normalization_records.append({
                            "entity_id": entity_id,
                            "ontology": ontology,
                        })

            elif record_type in {"#", "E"}:
                pass

            else:
                record_counts["unknown_record_type"] += 1

                unparsed_records.append({
                    "line": line_number,
                    "record_id": record_id,
                    "raw_line": line,
                })

        except (ValueError, IndexError) as exc:
            errors.append({
                "line": line_number,
                "record_id": record_id,
                "error": str(exc),
            })

    normalization_by_entity = defaultdict(set)

    for item in normalization_records:
        normalization_by_entity[item["entity_id"]].add(
            item["ontology"]
        )

    for entity in entities:
        entity["normalizations"] = sorted(
            normalization_by_entity.get(entity["entity_id"], set())
        )

    return {
        "entities": entities,
        "raw_labels": dict(raw_labels),
        "record_counts": dict(record_counts),
        "attribute_types": dict(attribute_types),
        "relation_types": dict(relation_types),
        "normalization_records": normalization_records,
        "offset_checks": dict(offset_checks),
        "errors": errors,
        "surface_mismatch_examples": surface_mismatch_examples,
        "unparsed_records": unparsed_records,
    }


def analyze_split(
    *,
    split: str,
    documents: list[Any],
    text_index: dict[str, Path],
    ann_index: dict[str, Path],
) -> dict[str, Any]:
    label_mentions = Counter()
    label_documents = defaultdict(set)
    continuous_mentions = Counter()
    discontinuous_mentions = Counter()
    normalization_entities = Counter()
    normalization_ontologies = defaultdict(Counter)

    raw_labels = Counter()
    record_counts = Counter()
    attribute_types = Counter()
    relation_types = Counter()
    offset_checks = Counter()

    doc_presence = {
        "both_ADR_and_Symptom": 0,
        "ADR_only": 0,
        "Symptom_only": 0,
        "neither_ADR_nor_Symptom": 0,
    }

    cross_label_overlap_pairs = 0
    exact_cross_label_span_pairs = 0

    parse_errors = []
    surface_mismatch_examples = []
    unparsed_records = []
    overlap_examples = []
    missing_files = []

    for document in documents:
        document_id = document.document_id

        text_path = text_index.get(document_id)
        ann_path = ann_index.get(document_id)

        if text_path is None or ann_path is None:
            missing_files.append(document_id)
            continue

        # Preserve original newline bytes for BRAT offsets.
        source_text = text_path.read_bytes().decode("utf-8")

        parsed = parse_ann_file(ann_path, source_text)
        surface_mismatch_examples.extend(
            {
                "document_id": document_id,
                **item,
            }
            for item in parsed["surface_mismatch_examples"]
        )

        unparsed_records.extend(
            {
                "document_id": document_id,
                **item,
            }
            for item in parsed["unparsed_records"]
        )
        for err in parsed["errors"]:
            parse_errors.append({
                "document_id": document_id,
                **err,
            })

        raw_labels.update(parsed["raw_labels"])
        record_counts.update(parsed["record_counts"])
        attribute_types.update(parsed["attribute_types"])
        relation_types.update(parsed["relation_types"])
        offset_checks.update(parsed["offset_checks"])

        present_labels = set()
        adr_mentions = []
        symptom_mentions = []

        for entity in parsed["entities"]:
            label = entity["canonical_label"]
            raw_label = entity["raw_label"]

            label_mentions[label] += 1
            label_documents[label].add(document_id)

            if entity["discontinuous"]:
                discontinuous_mentions[label] += 1
            else:
                continuous_mentions[label] += 1

            if entity["normalizations"]:
                normalization_entities[label] += 1

                normalization_ontologies[label].update(
                    entity["normalizations"]
                )

            if label != "UNKNOWN":
                present_labels.add(label)

            if label == "ADR":
                adr_mentions.append(entity)

            elif label == "Symptom":
                symptom_mentions.append(entity)

        has_adr = "ADR" in present_labels
        has_symptom = "Symptom" in present_labels

        if has_adr and has_symptom:
            doc_presence["both_ADR_and_Symptom"] += 1
        elif has_adr:
            doc_presence["ADR_only"] += 1
        elif has_symptom:
            doc_presence["Symptom_only"] += 1
        else:
            doc_presence["neither_ADR_nor_Symptom"] += 1

        # Check ADR/Symptom span overlap within the same document only.
        for adr in adr_mentions:
            for symptom in symptom_mentions:
                for a_start, a_end in adr["segments"]:
                    for s_start, s_end in symptom["segments"]:
                        overlaps = (
                            max(a_start, s_start) < min(a_end, s_end)
                        )
                        if overlaps and len(overlap_examples) < 200:
                            overlap_examples.append({
                                "document_id": document_id,
                                "adr": {
                                    "entity_id": adr["entity_id"],
                                    "surface": adr["surface"],
                                    "segments": adr["segments"],
                                },
                                "symptom": {
                                    "entity_id": symptom["entity_id"],
                                    "surface": symptom["surface"],
                                    "segments": symptom["segments"],
                                },
                                "overlap_start": max(a_start, s_start),
                                "overlap_end": min(a_end, s_end),
                                "exact_segment_match": (
                                    a_start == s_start and a_end == s_end
                                ),
                            })
                        if overlaps:
                            cross_label_overlap_pairs += 1

                            if (
                                a_start == s_start
                                and a_end == s_end
                            ):
                                exact_cross_label_span_pairs += 1

    labels_report = {}

    all_labels = sorted(
        set(label_mentions)
        | set(label_documents)
        | set(continuous_mentions)
        | set(discontinuous_mentions)
    )

    for label in all_labels:
        labels_report[label] = {
            "mentions": label_mentions[label],
            "documents": len(label_documents[label]),
            "continuous_mentions": continuous_mentions[label],
            "discontinuous_mentions": discontinuous_mentions[label],
            "normalized_entities": normalization_entities[label],
            "normalization_ontologies": dict(
                normalization_ontologies[label]
            ),
        }

    return {
        "split": split,
        "documents": len(documents) - len(missing_files),
        "labels": labels_report,
        "raw_label_counts": dict(raw_labels),
        "document_cooccurrence": doc_presence,
        "adr_symptom_segment_overlap_pairs": cross_label_overlap_pairs,
        "adr_symptom_exact_span_pairs": exact_cross_label_span_pairs,
        "annotation_record_types": dict(record_counts),
        "attribute_types": dict(attribute_types),
        "relation_types": dict(relation_types),
        "offset_checks": dict(offset_checks),
        "missing_files": missing_files,
        "parse_errors": parse_errors,
        "surface_mismatch_examples": surface_mismatch_examples,
        "unparsed_records": unparsed_records,
        "overlap_examples": overlap_examples,
    }


def debug_report(
    *,
    split_validation: dict[str, Any],
    split_reports: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "name": name,
            "level": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    add(
        "official_split_validation",
        bool(split_validation.get("valid")),
        str(split_validation.get("split_counts", {})),
    )

    for split, report in split_reports.items():
        add(
            f"{split}_document_count",
            report["documents"] == EXPECTED_SPLITS[split],
            f"{report['documents']} / {EXPECTED_SPLITS[split]}",
        )

        add(
            f"{split}_annotation_files",
            len(report["missing_files"]) == 0,
            f"missing={len(report['missing_files'])}",
        )

        add(
            f"{split}_annotation_parse",
            len(report["parse_errors"]) == 0,
            f"errors={len(report['parse_errors'])}",
        )

        offsets = report["offset_checks"]

        add(
            f"{split}_offset_bounds",
            offsets.get("out_of_bounds", 0) == 0,
            f"out_of_bounds={offsets.get('out_of_bounds', 0)}",
        )

    return {
        "data_root": str(data_root),
        "test_annotations_read": False,
        "model_inference_performed": False,
        "checks": checks,
        "fail_count": sum(
            item["level"] == "FAIL" for item in checks
        ),
        "passed": all(
            item["level"] == "PASS" for item in checks
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        adapter = CadecV2Adapter(args.data_root)
        split_validation = adapter.validate_splits()

        if not split_validation.get("valid"):
            raise ValueError(
                f"Official split validation failed: {split_validation}"
            )

        text_index = index_files(args.data_root / "txt", ".txt")
        ann_index = index_files(args.data_root / "ann", ".ann")

        split_reports = {}

        for split in ("train", "dev"):
            print(f"\nAuditing {split} annotations...")

            # The adapter is used only to obtain official split documents.
            # All raw labels are read directly from their .ann files.
            documents = adapter.load_split(
                split,
                labels={"Symptom"},
                include_discontinuous=True,
            )

            if len(documents) != EXPECTED_SPLITS[split]:
                raise ValueError(
                    f"{split}: loaded {len(documents)} documents; "
                    f"expected {EXPECTED_SPLITS[split]}"
                )

            report = analyze_split(
                split=split,
                documents=documents,
                text_index=text_index,
                ann_index=ann_index,
            )

            split_reports[split] = report

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        output_dir = (
            args.output_root
            / f"cadec_v2_annotation_audit_{stamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        summary = {
            "benchmark": "CADECv2",
            "audit_type": "offline_original_annotation_audit",
            "target_scope": ["ADR", "Symptom"],
            "all_original_labels_preserved": True,
            "audited_splits": ["train", "dev"],
            "test_annotations_read": False,
            "model_inference_performed": False,
            "split_validation": split_validation,
            "splits": split_reports,
        }

        debug = debug_report(
            split_validation=split_validation,
            split_reports=split_reports,
            data_root=args.data_root,
        )

        summary_path = output_dir / "annotation_audit.json"
        debug_path = output_dir / "debug_report.json"

        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        debug_path.write_text(
            json.dumps(debug, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\n========== CADECv2 ANNOTATION AUDIT ==========")

        for split, report in split_reports.items():
            print(f"\n--- {split.upper()} ---")
            print(f"Documents: {report['documents']}")

            for label, stats in sorted(report["labels"].items()):
                print(
                    f"{label:10s} "
                    f"mentions={stats['mentions']:6d} "
                    f"docs={stats['documents']:5d} "
                    f"continuous={stats['continuous_mentions']:6d} "
                    f"discontinuous={stats['discontinuous_mentions']:5d}"
                )

            print(
                "ADR/Symptom co-occurrence:",
                report["document_cooccurrence"],
            )

            print(
                "ADR/Symptom overlapping segment pairs:",
                report["adr_symptom_segment_overlap_pairs"],
            )

            print("Relations:", report["relation_types"])
            print("Attributes:", report["attribute_types"])

        print("\n========== DEBUG ==========")
        for check in debug["checks"]:
            print(
                f"[{check['level']}] {check['name']}: "
                f"{check['detail']}"
            )

        print(f"\nAudit JSON: {summary_path}")
        print(f"Debug JSON: {debug_path}")

        return 0 if debug["passed"] else 2

    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
