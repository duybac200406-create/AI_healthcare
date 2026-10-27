#!/usr/bin/env python3
"""Prepare a blinded, stratified 30-item PubMedQA PQA-L feasibility pilot.

No third-party libraries are required. By default, downloads the public, official
PQA-L JSON from the PubMedQA GitHub repository and tries to download the official
500-item test-key file so those IDs can be excluded from the pilot sample.

Outputs:
  pubmedqa_pilot_30_blinded.csv  - annotate this BEFORE opening the key
  pubmedqa_pilot_30_key.json     - gold-label mapping; keep separate while annotating

The pilot is for feasibility only, not a statistically powered evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

DATA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
TEST_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 PubMedQA-feasibility-pilot"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
        dest.write_bytes(content)
        print(f"Downloaded: {dest} ({len(content):,} bytes)")
        return True
    except Exception as exc:  # network availability varies by institution / region
        print(f"WARNING: Could not download {url}: {exc}", file=sys.stderr)
        return False


def get_ci(obj: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Get a dictionary field without depending on key capitalization."""
    lookup = {str(k).lower(): v for k, v in obj.items()}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return default


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def build_context(entry: dict[str, Any]) -> str:
    """Build evidence from context passages only; never include LONG_ANSWER."""
    contexts = get_ci(entry, "CONTEXTS", "contexts")
    if isinstance(contexts, dict):
        # Hugging Face-like nested schema. The label array is metadata, not
        # necessarily a one-to-one alignment with context blocks, so do not
        # attach it to individual evidence passages.
        contexts = contexts.get("contexts", [])

    context_list = as_string_list(contexts)
    if not context_list:
        # Some mirrors represent the entire context as a single string.
        context_list = as_string_list(get_ci(entry, "context", default=""))

    blocks = []
    for i, passage in enumerate(context_list, start=1):
        # The source LABELS field is not guaranteed to align one-to-one with
        # CONTEXTS; use neutral evidence IDs only. Do not split on punctuation,
        # which can split medical abbreviations incorrectly.
        blocks.append(f"[E{i}]\n{passage}")
    return "\n\n".join(blocks).strip()


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {"yes": "yes", "no": "no", "maybe": "maybe"}
    return mapping.get(text, "")


def iter_records(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                rows.append((str(key), value))
    elif isinstance(raw, list):
        for idx, value in enumerate(raw):
            if isinstance(value, dict):
                pubid = get_ci(value, "pubid", "pubid", "PMID", default=f"row-{idx}")
                rows.append((str(pubid), value))
    else:
        raise ValueError("Unrecognized JSON structure: expected a dict or list.")
    return rows


def load_test_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not parse test-key file {path}: {exc}", file=sys.stderr)
        return set()
    ids: set[str] = set()
    if isinstance(raw, dict):
        ids.update(str(k) for k in raw.keys())
        # If nested, also look for common identifier fields.
        for v in raw.values():
            if isinstance(v, dict):
                pid = get_ci(v, "pubid", "PMID", "id")
                if pid is not None:
                    ids.add(str(pid))
    elif isinstance(raw, list):
        for v in raw:
            if isinstance(v, (str, int)):
                ids.add(str(v))
            elif isinstance(v, dict):
                pid = get_ci(v, "pubid", "PMID", "id")
                if pid is not None:
                    ids.add(str(pid))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ori_pqal.json"), help="Path to official PQA-L JSON")
    parser.add_argument("--test-key", type=Path, default=Path("test_ground_truth.json"), help="Path to official test IDs/labels")
    parser.add_argument("--outdir", type=Path, default=Path("pubmedqa_pilot"), help="Output directory")
    parser.add_argument("--per-label", type=int, default=10, help="Number sampled per gold class (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}. Trying the official public raw URL...")
        if not download(DATA_URL, args.data):
            print("\nDownload failed. Manually download ori_pqal.json from:\n"
                  "https://github.com/pubmedqa/pubmedqa/tree/master/data\n"
                  "then rerun with --data PATH_TO_ori_pqal.json", file=sys.stderr)
            return 2

    if not args.test_key.exists():
        download(TEST_URL, args.test_key)

    try:
        raw = json.loads(args.data.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read JSON file {args.data}: {exc}", file=sys.stderr)
        return 2

    test_ids = load_test_ids(args.test_key)
    if not test_ids:
        print("WARNING: Official test IDs were not loaded. The pilot may overlap the reserved test set; treat this only as a feasibility sample and do not use these IDs as final evaluation data.", file=sys.stderr)
    else:
        print(f"Loaded {len(test_ids)} IDs from the official test-key file; these will be excluded.")

    records: list[dict[str, Any]] = []
    for fallback_id, entry in iter_records(raw):
        pubid = str(get_ci(entry, "pubid", "PMID", default=fallback_id))
        label = normalize_label(get_ci(entry, "final_decision", "FINAL_DECISION", "label"))
        question = str(get_ci(entry, "QUESTION", "question", default="") or "").strip()
        long_answer = str(get_ci(entry, "LONG_ANSWER", "long_answer", default="") or "").strip()
        context = build_context(entry)
        if not label or not question or not context:
            continue
        if pubid in test_ids:
            continue
        records.append({
            "pubid": pubid,
            "label": label,
            "question": question,
            "context": context,
            "long_answer": long_answer,
        })

    by_label = {label: [r for r in records if r["label"] == label] for label in ("yes", "no", "maybe")}
    print("Available non-test candidates by class:", {k: len(v) for k, v in by_label.items()})
    short = {k: len(v) for k, v in by_label.items() if len(v) < args.per_label}
    if short:
        print(f"ERROR: Not enough candidates for requested stratified sample ({short}). Change --per-label or inspect dataset/test-key parsing.", file=sys.stderr)
        return 3

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for label in ("yes", "no", "maybe"):
        pool = list(by_label[label])
        rng.shuffle(pool)
        selected.extend(pool[:args.per_label])
    rng.shuffle(selected)

    args.outdir.mkdir(parents=True, exist_ok=True)
    blinded_path = args.outdir / f"pubmedqa_pilot_{len(selected)}_blinded.csv"
    key_path = args.outdir / f"pubmedqa_pilot_{len(selected)}_key.json"

    annotation_fields = [
        "evidence_sufficiency_label",
        "supporting_or_missing_information_note",
        "conflict_or_ambiguity_note",
        "annotator_confidence_1_to_3",
        "second_annotator_label",
        "adjudicated_label",
    ]
    key = {}
    with blinded_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["pilot_id", "question", "evidence_context", *annotation_fields])
        writer.writeheader()
        for i, r in enumerate(selected, start=1):
            pilot_id = f"P{i:03d}"
            writer.writerow({
                "pilot_id": pilot_id,
                "question": r["question"],
                "evidence_context": r["context"],
                **{field: "" for field in annotation_fields},
            })
            key[pilot_id] = {
                "pubid": r["pubid"],
                "gold_final_decision": r["label"],
                "gold_long_answer": r["long_answer"],
            }

    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPilot files created:")
    print(f"  Blinded annotation file: {blinded_path.resolve()}")
    print(f"  GOLD KEY (keep closed until sufficiency labels are frozen): {key_path.resolve()}")
    print("\nImportant: this sample is for feasibility, not final evaluation. Preserve the key separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
