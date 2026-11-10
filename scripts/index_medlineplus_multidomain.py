"""Index the captured MedlinePlus multi-domain pilot into a dedicated Qdrant collection.

This script deliberately validates all source JSON files before writing any points.
It is idempotent per document because DocumentIndexer/QdrantVectorStore use
stable document, chunk, and pipeline identities.
"""

from __future__ import annotations
import argparse
import httpx
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.providers.ollama_embedding_provider import (
    DEFAULT_EMBEDDING_MODEL,
    OllamaEmbeddingProvider,
)
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "samples" / "medlineplus_health_topics"
EXPECTED_DOCUMENT_COUNT = 7
EXPECTED_DOMAINS = frozenset(
    {
        "cardiovascular",
        "endocrine_metabolic",
        "respiratory",
        "gastrointestinal",
        "renal",
        "neurology",
        "mental_health",
    }
)
BASELINE_COLLECTION_NAME = (
    "medical_knowledge_bge_m3_multidomain_pilot_v1"
)
QDRANT_URL = "http://127.0.0.1:6333"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL  # bge-m3; matches the existing provider/store convention.
VECTOR_SIZE = 1024


class SourceValidationError(RuntimeError):
    """Raised when the captured source corpus fails pre-index validation."""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index captured MedlinePlus documents into a new Qdrant collection."
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Name of a new Qdrant collection. Existing collections are never overwritten.",
    )

    args = parser.parse_args()

    if not args.collection_name.strip():
        parser.error("--collection-name must not be blank.")

    if args.collection_name == BASELINE_COLLECTION_NAME:
        parser.error(
            "Refusing to overwrite the existing baseline collection."
        )

    return args


def assert_collection_does_not_exist(collection_name: str) -> None:
    """Fail closed if Qdrant cannot confirm the target collection is absent."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{QDRANT_URL}/collections")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceValidationError(
            "Cannot verify Qdrant collections. Refusing to index."
        ) from exc

    collections = payload.get("result", {}).get("collections")

    if not isinstance(collections, list):
        raise SourceValidationError(
            "Unexpected Qdrant collection response. Refusing to index."
        )

    existing_names = {
        item.get("name")
        for item in collections
        if isinstance(item, dict)
    }

    if collection_name in existing_names:
        raise SourceValidationError(
            f"Target collection already exists: {collection_name}. "
            "Refusing to overwrite it."
        )
def _load_source_documents(source_dir: Path) -> tuple[MedicalDocument, ...]:
    if not source_dir.is_dir():
        raise SourceValidationError(f"Source directory does not exist: {source_dir}")

    files = sorted(source_dir.glob("*.json"))
    if len(files) != EXPECTED_DOCUMENT_COUNT:
        raise SourceValidationError(
            f"Expected exactly {EXPECTED_DOCUMENT_COUNT} JSON source files; found {len(files)}."
        )

    documents: list[MedicalDocument] = []
    seen_document_ids: set[str] = set()

    for path in files:
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            document = MedicalDocument.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise SourceValidationError(
                f"Invalid captured source file '{path.name}': {type(exc).__name__}."
            ) from exc

        if document.document_id in seen_document_ids:
            raise SourceValidationError(
                f"Duplicate document_id in source corpus: {document.document_id}."
            )
        seen_document_ids.add(document.document_id)

        if document.source_type != "health_topic_summary":
            raise SourceValidationError(
                f"Unexpected source_type for '{document.document_id}'."
            )
        if document.source_url is None:
            raise SourceValidationError(
                f"Missing source URL for '{document.document_id}'."
            )
        if document.source_url.scheme != "https" or document.source_url.host != "medlineplus.gov":
            raise SourceValidationError(
                f"Source URL must use HTTPS on medlineplus.gov for '{document.document_id}'."
            )
        if document.source_name != "MedlinePlus, National Library of Medicine":
            raise SourceValidationError(
                f"Unexpected source organization for '{document.document_id}'."
            )
        if document.source_last_updated is None:
            raise SourceValidationError(
                f"Missing source_last_updated for '{document.document_id}'."
            )
        if document.publication_date is not None:
            raise SourceValidationError(
                f"publication_date must remain unset unless a publication date is verified: {document.document_id}."
            )
        if document.language != "en":
            raise SourceValidationError(
                f"Expected English source text for '{document.document_id}'."
            )
        if not document.license or "public domain" not in document.license.casefold():
            raise SourceValidationError(
                f"Missing expected public-domain attribution metadata for '{document.document_id}'."
            )
        if len(document.content.strip()) < 500:
            raise SourceValidationError(
                f"FullSummary content is unexpectedly short for '{document.document_id}'."
            )

        documents.append(document)

    actual_domains = frozenset(document.medical_domain for document in documents)
    if actual_domains != EXPECTED_DOMAINS:
        missing = sorted(EXPECTED_DOMAINS - actual_domains)
        unexpected = sorted(actual_domains - EXPECTED_DOMAINS)
        raise SourceValidationError(
            f"Medical-domain coverage mismatch. Missing={missing}; unexpected={unexpected}."
        )

    if len({document.medical_domain for document in documents}) != EXPECTED_DOCUMENT_COUNT:
        raise SourceValidationError("Pilot must contain exactly one document per expected domain.")

    return tuple(documents)


def main() -> int:
    args = parse_args()
    collection_name = args.collection_name

    print("=== MEDLINEPLUS MULTI-DOMAIN INDEXING ===")
    print("SOURCE DIRECTORY:", SOURCE_DIR)
    print("TARGET COLLECTION:", collection_name)
    print("EMBEDDING MODEL:", EMBEDDING_MODEL)
    print("VECTOR DIMENSIONS:", VECTOR_SIZE)

    # Validate every source before opening the write path to Qdrant.
    try:
        assert_collection_does_not_exist(collection_name)
        documents = _load_source_documents(SOURCE_DIR)
    except SourceValidationError as exc:
        print("SOURCE VALIDATION FAILED:", exc, file=sys.stderr)
        return 1

    print(f"SOURCE VALIDATION: PASS ({len(documents)} documents, {len(EXPECTED_DOMAINS)} domains)")
    for document in documents:
        print(
            f"VALID: {document.document_id} | {document.medical_domain} | "
            f"{document.source_last_updated} | {len(document.content)} chars"
        )

    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=VECTOR_SIZE,
        distance="cosine",
        embedding_model=EMBEDDING_MODEL,
    )

    indexed: list[tuple[MedicalDocument, str, int]] = []
    try:
        with OllamaEmbeddingProvider(model=EMBEDDING_MODEL, timeout=180.0) as embedding_provider:
            with QdrantVectorStore(url=QDRANT_URL, config=config, timeout=30.0) as vector_store:
                vector_store.ensure_collection()
                indexer = DocumentIndexer(
                    embedding_provider=embedding_provider,
                    vector_store=vector_store,
                    max_chars=1200,
                    overlap_chars=150,
                    batch_size=16,
                )

                for document in documents:
                    ingestion = indexer.index_document(document)
                    document_count = vector_store.count(
                        VectorSearchFilter(document_ids=(document.document_id,))
                    )
                    if document_count != ingestion.chunk_count:
                        raise RuntimeError(
                            f"Active chunk count mismatch for {document.document_id}: "
                            f"expected {ingestion.chunk_count}, got {document_count}."
                        )
                    indexed.append(
                        (document, ingestion.pipeline_fingerprint, ingestion.chunk_count)
                    )
                    print(
                        f"INDEXED: {document.document_id} | domain={document.medical_domain} | "
                        f"chunks={ingestion.chunk_count} | "
                        f"fingerprint={ingestion.pipeline_fingerprint[:12]}"
                    )

                print("\n=== ACTIVE QDRANT COUNTS ===")
                total = 0
                for domain in sorted(EXPECTED_DOMAINS):
                    count = vector_store.count(
                        VectorSearchFilter(medical_domains=(domain,))
                    )
                    total += count
                    print(f"{domain}: {count} chunks")

                expected_total = sum(chunk_count for _, _, chunk_count in indexed)
                if total != expected_total:
                    raise RuntimeError(
                        f"Collection count mismatch: expected {expected_total}, got {total}."
                    )
                print("TOTAL ACTIVE CHUNKS:", total)
                print("INDEXING STATUS: PASS")
                print("COLLECTION:", collection_name)
                print("NOTE: This is a 7-document pilot corpus, not a validated comprehensive medical knowledge base.")

    except Exception as exc:
        print(
            "INDEXING FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if indexed:
            print(
                f"PARTIAL INDEXING: {len(indexed)}/{len(documents)} documents had completed. "
                "Rerun the script after resolving the error; indexing is idempotent per document.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
