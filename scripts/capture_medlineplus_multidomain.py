from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

import httpx
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_healthcare.ingestion.medlineplus_capture import (  # noqa: E402
    MedlinePlusCaptureError,
    capture_health_topic,
)
from ai_healthcare.schemas.document import MedicalDocument  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "medlineplus_multidomain_pilot_v0.2.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "samples" / "medlineplus_health_topics"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture exact MedlinePlus Health Topic FullSummary records "
            "through the official NLM Web Service."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest JSON. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for immutable source JSON snapshots.",
    )
    parser.add_argument(
        "--document-ids",
        nargs="+",
        default=None,
        help="Capture only these document IDs. Omit to select every manifest source.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly replace existing snapshot files after fetching and validating.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.timeout <= 0 or args.delay_seconds < 0:
        print("ERROR: timeout must be positive and delay-seconds cannot be negative.")
        return 2

    manifest_path = _repo_path(args.manifest)
    output_dir = _repo_path(args.output_dir)

    # Protect the original seven-document snapshot directory from accidental
    # use with a different/expanded manifest.
    if (
        output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        and manifest_path.resolve() != DEFAULT_MANIFEST.resolve()
    ):
        print(
            "ERROR: refusing to use the baseline output directory with a "
            "different manifest. Choose a new --output-dir."
        )
        return 2

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read manifest: {type(exc).__name__}: {exc}")
        return 2

    entries = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        print("ERROR: manifest must contain a non-empty 'sources' list.")
        return 2

    entries_by_id: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            print("ERROR: every manifest source must be a JSON object.")
            return 2
        document_id = entry.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            print("ERROR: every manifest source requires a non-empty document_id.")
            return 2
        if document_id in entries_by_id:
            print(f"ERROR: duplicate document_id in manifest: {document_id}")
            return 2
        entries_by_id[document_id] = entry

    if args.document_ids:
        requested_ids = list(dict.fromkeys(args.document_ids))
        missing_ids = sorted(set(requested_ids) - set(entries_by_id))
        if missing_ids:
            print(f"ERROR: requested document IDs are absent from manifest: {missing_ids}")
            return 2
        selected_entries = [entries_by_id[document_id] for document_id in requested_ids]
    else:
        selected_entries = entries

    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    headers = {
        "User-Agent": "ai-healthcare-medlineplus-capture/1.0 (NLM source attribution retained)",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
    }

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:
        for index, spec in enumerate(selected_entries):
            document_id = spec["document_id"]
            output_path = output_dir / f"{document_id}.json"

            # Never silently replace an existing source snapshot. A fresh and
            # metadata-matching file may be reused; all other replacements
            # require the explicit --refresh flag.
            if output_path.exists() and not args.refresh:
                try:
                    existing = MedicalDocument.model_validate_json(
                        output_path.read_text(encoding="utf-8")
                    )
                    age = datetime.now(timezone.utc) - existing.retrieved_at
                    source_updated = (
                        existing.source_last_updated.isoformat()
                        if existing.source_last_updated is not None
                        else None
                    )
                    metadata_matches = (
                        existing.document_id == document_id
                        and str(existing.source_url) == str(spec.get("url"))
                        and existing.title == spec.get("title")
                        and existing.medical_domain == spec.get("medical_domain")
                        and source_updated == spec.get("source_last_updated")
                    )
                    cache_is_fresh = timedelta(0) <= age <= timedelta(hours=24)

                    if cache_is_fresh and metadata_matches:
                        print(f"CACHED: {document_id} ({output_path.relative_to(ROOT)})")
                        successes += 1
                        continue
                except (ValidationError, ValueError, OSError):
                    pass

                print(
                    f"EXISTS BUT NOT REUSED: {document_id}. "
                    "Snapshot was not overwritten; inspect it or use --refresh explicitly."
                )
                failures += 1
                continue

            try:
                document = capture_health_topic(spec, client=client)
                temp_path = output_path.with_suffix(".json.tmp")
                temp_path.write_text(
                    json.dumps(
                        document.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temp_path.replace(output_path)

                print(
                    f"CAPTURED: {document.document_id} | {document.title} | "
                    f"{document.medical_domain} | {document.source_last_updated} | "
                    f"{len(document.content)} chars"
                )
                successes += 1
            except (MedlinePlusCaptureError, OSError, ValueError) as exc:
                print(f"FAILED: {document_id} | {type(exc).__name__}: {exc}")
                failures += 1

            if index < len(selected_entries) - 1 and args.delay_seconds:
                time.sleep(args.delay_seconds)

    print(f"\nSOURCE CAPTURE SUMMARY: {successes} succeeded, {failures} failed")
    print("MANIFEST:", manifest_path)
    print("OUTPUT:", output_dir)
    print("POLICY: exact MedlinePlus FullSummary only; no third-party linked content.")
    print("SNAPSHOT POLICY: existing files are never replaced unless --refresh is explicit.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
