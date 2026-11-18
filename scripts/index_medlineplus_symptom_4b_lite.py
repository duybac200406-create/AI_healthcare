"""Index the MedlinePlus symptom-focused 4B-lite corpus into a NEW Qdrant collection.

Differences from index_medlineplus_multidomain.py (which is left untouched):
* reads data/samples/medlineplus_symptom_4b_lite_v1 (14 documents);
* validates the documents against the 4B-lite manifest instead of a hard-coded
  "one document per domain" rule, because several domains now hold more than
  one document and two documents carry the explicit domain "other";
* offers --validate-only, which checks the files without contacting Qdrant or
  Ollama.

Existing collections are never overwritten.

Usage (PowerShell, from the repository root):
    python scripts/index_medlineplus_symptom_4b_lite.py --validate-only
    python scripts/index_medlineplus_symptom_4b_lite.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.ingestion.structured_indexer import StructuredDocumentIndexer
from ai_healthcare.providers.ollama_embedding_provider import (
    DEFAULT_EMBEDDING_MODEL,
    OllamaEmbeddingProvider,
)
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "samples" / "medlineplus_symptom_4b_lite_v1"
MANIFEST_PATH = ROOT / "data" / "manifests" / "medlineplus_symptom_4b_lite_v1.json"

DEFAULT_COLLECTION_NAMES = {
    "structured": "medical_knowledge_bge_m3_symptom_4b_lite_structured_v1",
    "char": "medical_knowledge_bge_m3_symptom_4b_lite_plaintext_v1",
}
PROTECTED_COLLECTION_NAMES = frozenset(
    {
        "medical_knowledge_bge_m3_multidomain_pilot_v1",
        "medical_knowledge_bge_m3_multidomain_plaintext_v1",
    }
)
QDRANT_URL = "http://127.0.0.1:6333"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
VECTOR_SIZE = 1024
MIN_CONTENT_CHARS = 500


class SourceValidationError(RuntimeError):
    """Raised when the captured corpus fails pre-index validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--chunker",
        choices=("structured", "char"),
        default="structured",
        help="structured = section/sentence-aware (default); "
        "char = original 1200/150 character chunker.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Name of a NEW collection (default depends on --chunker).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the source files and manifest, then exit.",
    )
    args = parser.parse_args()

    if args.collection_name is None:
        args.collection_name = DEFAULT_COLLECTION_NAMES[args.chunker]
    if not args.collection_name.strip():
        parser.error("--collection-name must not be blank.")
    if args.collection_name in PROTECTED_COLLECTION_NAMES:
        parser.error("Refusing to use a protected existing collection name.")
    return args


def assert_collection_does_not_exist(collection_name: str) -> None:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{QDRANT_URL}/collections")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceValidationError(
            f"Cannot reach Qdrant at {QDRANT_URL}. Is it running? "
            "Refusing to index."
        ) from exc

    collections = payload.get("result", {}).get("collections")
    if not isinstance(collections, list):
        raise SourceValidationError(
            "Unexpected Qdrant collection response. Refusing to index."
        )
    names = {c.get("name") for c in collections if isinstance(c, dict)}
    if collection_name in names:
        raise SourceValidationError(
            f"Target collection already exists: {collection_name}. "
            "Choose a different --collection-name."
        )


def _load_manifest() -> dict[str, dict[str, Any]]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceValidationError(
            f"Cannot read manifest: {MANIFEST_PATH}"
        ) from exc

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourceValidationError("Manifest has no 'sources' list.")

    by_id: dict[str, dict[str, Any]] = {}
    for item in sources:
        doc_id = item.get("document_id")
        if not doc_id or doc_id in by_id:
            raise SourceValidationError(
                f"Missing or duplicate document_id in manifest: {doc_id!r}."
            )
        by_id[doc_id] = item
    return by_id


def load_source_documents() -> tuple[MedicalDocument, ...]:
    if not SOURCE_DIR.is_dir():
        raise SourceValidationError(f"Source directory missing: {SOURCE_DIR}")

    manifest = _load_manifest()
    files = sorted(SOURCE_DIR.glob("*.json"))
    if len(files) != len(manifest):
        raise SourceValidationError(
            f"Manifest lists {len(manifest)} documents; found {len(files)} files."
        )

    documents: list[MedicalDocument] = []
    seen: set[str] = set()

    for path in files:
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            doc = MedicalDocument.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise SourceValidationError(
                f"Invalid source file '{path.name}': {type(exc).__name__}."
            ) from exc

        if doc.document_id in seen:
            raise SourceValidationError(f"Duplicate document_id: {doc.document_id}.")
        seen.add(doc.document_id)

        entry = manifest.get(doc.document_id)
        if entry is None:
            raise SourceValidationError(
                f"{doc.document_id} is not listed in the manifest."
            )
        if entry.get("medical_domain") != doc.medical_domain:
            raise SourceValidationError(
                f"Domain mismatch for {doc.document_id}: manifest="
                f"{entry.get('medical_domain')!r}, file={doc.medical_domain!r}."
            )

        problems: list[str] = []
        if doc.source_type != "health_topic_summary":
            problems.append("unexpected source_type")
        if doc.source_url is None:
            problems.append("missing source URL")
        elif doc.source_url.scheme != "https" or doc.source_url.host != "medlineplus.gov":
            problems.append("source URL must be HTTPS on medlineplus.gov")
        if doc.source_name != "MedlinePlus, National Library of Medicine":
            problems.append("unexpected source organization")
        if doc.source_last_updated is None:
            problems.append("missing source_last_updated")
        if doc.publication_date is not None:
            problems.append("publication_date must stay unset")
        if doc.language != "en":
            problems.append("expected English text")
        if not doc.license or "public domain" not in doc.license.casefold():
            problems.append("missing public-domain attribution")
        if len(doc.content.strip()) < MIN_CONTENT_CHARS:
            problems.append(f"content shorter than {MIN_CONTENT_CHARS} characters")
        if problems:
            raise SourceValidationError(
                f"{doc.document_id}: " + "; ".join(problems) + "."
            )
        documents.append(doc)

    missing = sorted(set(manifest) - seen)
    if missing:
        raise SourceValidationError(f"Manifest documents without files: {missing}.")
    return tuple(documents)


def main() -> int:
    args = parse_args()
    collection_name = args.collection_name

    print("=== MEDLINEPLUS SYMPTOM 4B-LITE INDEXING ===")
    print("SOURCE DIRECTORY:", SOURCE_DIR)
    print("TARGET COLLECTION:", collection_name)
    print("EMBEDDING MODEL:", EMBEDDING_MODEL)
    print("CHUNKER:", args.chunker)

    try:
        documents = load_source_documents()
        if not args.validate_only:
            assert_collection_does_not_exist(collection_name)
    except SourceValidationError as exc:
        print("SOURCE VALIDATION FAILED:", exc, file=sys.stderr)
        return 1

    domain_counts = Counter(d.medical_domain for d in documents)
    print(f"SOURCE VALIDATION: PASS ({len(documents)} documents)")
    for d in documents:
        print(
            f"VALID: {d.document_id} | {d.medical_domain} | "
            f"{d.source_last_updated} | {len(d.content)} chars"
        )
    print("DOMAINS:", dict(sorted(domain_counts.items())))

    if args.validate_only:
        print("VALIDATE-ONLY: nothing was written.")
        return 0

    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=VECTOR_SIZE,
        distance="cosine",
        embedding_model=EMBEDDING_MODEL,
    )

    indexed: list[tuple[MedicalDocument, int]] = []
    try:
        with OllamaEmbeddingProvider(model=EMBEDDING_MODEL, timeout=180.0) as embedder:
            with QdrantVectorStore(url=QDRANT_URL, config=config, timeout=30.0) as store:
                store.ensure_collection()
                if args.chunker == "structured":
                    indexer = StructuredDocumentIndexer(
                        embedding_provider=embedder,
                        vector_store=store,
                        max_chars=1200,
                        batch_size=16,
                    )
                else:
                    indexer = DocumentIndexer(
                        embedding_provider=embedder,
                        vector_store=store,
                        max_chars=1200,
                        overlap_chars=150,
                        batch_size=16,
                    )

                for doc in documents:
                    ingestion = indexer.index_document(doc)
                    count = store.count(
                        VectorSearchFilter(document_ids=(doc.document_id,))
                    )
                    if count != ingestion.chunk_count:
                        raise RuntimeError(
                            f"Chunk count mismatch for {doc.document_id}: "
                            f"expected {ingestion.chunk_count}, got {count}."
                        )
                    indexed.append((doc, ingestion.chunk_count))
                    print(
                        f"INDEXED: {doc.document_id} | domain={doc.medical_domain} "
                        f"| chunks={ingestion.chunk_count}"
                    )

                print("\n=== ACTIVE QDRANT COUNTS ===")
                total = 0
                for domain in sorted(domain_counts):
                    n = store.count(VectorSearchFilter(medical_domains=(domain,)))
                    total += n
                    print(f"{domain}: {n} chunks")

                expected = sum(c for _, c in indexed)
                if total != expected:
                    raise RuntimeError(
                        f"Collection count mismatch: expected {expected}, got {total}."
                    )
                print("TOTAL ACTIVE CHUNKS:", total)
                print("INDEXING STATUS: PASS")
                print("COLLECTION:", collection_name)
                print(
                    "NOTE: a 14-document pilot corpus, not a validated "
                    "comprehensive medical knowledge base."
                )
    except Exception as exc:
        print("INDEXING FAILED:", f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if indexed:
            print(
                f"PARTIAL: {len(indexed)}/{len(documents)} documents completed. "
                "Fix the error and rerun with a NEW --collection-name.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
