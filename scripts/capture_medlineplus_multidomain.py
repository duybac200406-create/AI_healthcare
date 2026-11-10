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

MANIFEST = ROOT / "data" / "manifests" / "medlineplus_multidomain_pilot_v0.2.json"
OUTPUT_DIR = ROOT / "data" / "samples" / "medlineplus_health_topics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture only MedlinePlus Health Topic Summary records via the official Web Service."
    )
    parser.add_argument("--refresh", action="store_true", help="Fetch again even when a validated local capture exists.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.delay_seconds < 0:
        print("ERROR: timeout must be positive and delay-seconds cannot be negative.")
        return 2
    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}")
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        print("ERROR: manifest must contain a non-empty 'sources' list.")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    headers = {
        "User-Agent": "ai-healthcare-multidomain-pilot/0.1 (MedlinePlus source attribution retained)",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
    }

    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        for index, spec in enumerate(entries):
            document_id = str(spec.get("document_id", f"source-{index + 1}"))
            output_path = OUTPUT_DIR / f"{document_id}.json"
            if output_path.exists() and not args.refresh:
                try:
                    existing = MedicalDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
                    age = datetime.now(timezone.utc) - existing.retrieved_at
                    cache_is_fresh = timedelta(0) <= age <= timedelta(hours=24)
                    metadata_matches = (
                        existing.document_id == document_id
                        and str(existing.source_url) == str(spec.get("url"))
                        and existing.title == spec.get("title")
                        and existing.medical_domain == spec.get("medical_domain")
                        and existing.source_last_updated.isoformat() == spec.get("source_last_updated")
                    )
                    if cache_is_fresh and metadata_matches:
                        print(f"CACHED: {document_id} ({output_path.relative_to(ROOT)})")
                        successes += 1
                        continue
                except (ValidationError, ValueError, AttributeError):
                    pass

            try:
                document = capture_health_topic(spec, client=client)
                temp_path = output_path.with_suffix(".json.tmp")
                temp_path.write_text(
                    json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
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

            if index < len(entries) - 1 and args.delay_seconds:
                time.sleep(args.delay_seconds)

    print(f"\nSOURCE CAPTURE SUMMARY: {successes} succeeded, {failures} failed")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print("POLICY: only MedlinePlus FullSummary; no A.D.A.M., AHFS, or third-party linked content.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
