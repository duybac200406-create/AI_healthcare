"""Compare the original character chunker with the structure-aware chunker.

Needs neither Qdrant nor Ollama. Reads the 4B-lite documents and reports,
per document and in total, how many chunk boundaries fall in the middle of a
line, which is where sentences get cut.

Usage:  python scripts/inspect_chunks.py [--show headache gerd]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION, chunk_document
from ai_healthcare.ingestion.normalizer import normalize_document
from ai_healthcare.ingestion.structured_chunker import (
    STRUCTURED_CHUNKER_VERSION,
    chunk_document_structured,
)
from ai_healthcare.schemas.document import MedicalDocument

SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data" / "samples" / "medlineplus_symptom_4b_lite_v1"
)


def unclean(source: str, chunks) -> tuple[int, int]:
    starts = sum(
        1 for c in chunks if c.start_char and source[c.start_char - 1] != "\n"
    )
    ends = sum(
        1 for c in chunks
        if c.end_char < len(source) and source[c.end_char] != "\n"
    )
    return starts, ends


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show", nargs="*", default=[],
                        help="document name fragments whose chunks to print")
    args = parser.parse_args()

    print(f"{'document':26s} {'old n':>6s} {'bad s/e':>8s} {'new n':>6s} {'bad s/e':>8s}")
    totals = [0, 0, 0, 0, 0, 0]
    for path in sorted(SOURCE_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc = normalize_document(MedicalDocument.model_validate(raw))
        old = chunk_document(doc, max_chars=1200, overlap_chars=150)
        new = chunk_document_structured(doc, max_chars=1200)
        os_, oe = unclean(doc.content, old)
        ns, ne = unclean(doc.content, new)
        name = doc.document_id.replace("doc_medlineplus_", "")
        print(f"{name:26s} {len(old):6d} {os_:>4d}/{oe:<3d} {len(new):6d} {ns:>4d}/{ne:<3d}")
        for i, v in enumerate((len(old), os_, oe, len(new), ns, ne)):
            totals[i] += v
        if any(fragment in name for fragment in args.show):
            for c in new:
                head = c.content.replace("\n", " | ")[:70]
                tail = c.content.replace("\n", " | ")[-50:]
                print(f"    [{c.chunk_index}] {len(c.content):4d} chars "
                      f"title={c.section_title!r}\n        {head} ... {tail}")

    print("-" * 56)
    print(f"{'TOTAL':26s} {totals[0]:6d} {totals[1]:>4d}/{totals[2]:<3d} "
          f"{totals[3]:6d} {totals[4]:>4d}/{totals[5]:<3d}")
    print(f"\nold = {CHUNKER_VERSION}   new = {STRUCTURED_CHUNKER_VERSION}")
    print("bad s/e = chunks that start / end in the middle of a line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
