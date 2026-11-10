"""Evaluate dense retrieval against the 35-case multi-domain pilot benchmark.

Runs BGE-M3 + Qdrant only. It does not call Qwen and does not evaluate clinical
answer correctness or abstention. Similarity scores are reported as similarities,
never converted to confidence or answerability decisions.
"""

from __future__ import annotations
from qdrant_client import QdrantClient
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_healthcare.evaluation.multidomain_retrieval import (
    aggregate_metrics,
    evaluate_ranked_results,
)
from ai_healthcare.providers.ollama_embedding_provider import (
    DEFAULT_EMBEDDING_MODEL,
    OllamaEmbeddingProvider,
)
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore

import hashlib

from ai_healthcare.ingestion.chunker import CHUNKER_VERSION
from ai_healthcare.ingestion.normalizer import (
    NORMALIZER_VERSION,
    normalize_text,
)
from ai_healthcare.schemas.ingestion import (
    IngestionRecord,
    PIPELINE_FINGERPRINT_VERSION,
)
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "evaluation" / "medical_rag_multidomain_questions_v0.1.json"
MANIFEST_PATH = ROOT / "data" / "manifests" / "medlineplus_multidomain_pilot_v0.2.json"
COLLECTION_NAME = "medical_knowledge_bge_m3_multidomain_pilot_v1"
QDRANT_URL = "http://127.0.0.1:6333"
VECTOR_SIZE = 1024


class EvaluationInputError(RuntimeError):
    """Raised when benchmark/manifest cannot be safely evaluated."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"Cannot read {path.name}: {type(exc).__name__}.") from exc
    if not isinstance(data, dict):
        raise EvaluationInputError(f"{path.name} must contain a JSON object.")
    return data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(data: Any) -> str:
    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_provenance(
    provenance: dict[str, Any],
    benchmark: dict[str, Any],
    manifest: dict[str, Any],
    collection_name: str,
) -> dict[str, str]:
    """Validate a retrospectively reconstructed index provenance artifact.

    This validates source files, fingerprints, configuration and
    artifact lineage. Live Qdrant payload validation is separate.
    """
    try:
        # 1. Validate artifact integrity.
        claimed_hash = provenance.get("provenance_payload_sha256")

        if not isinstance(claimed_hash, str):
            raise EvaluationInputError(
                "Provenance artifact has no payload SHA-256."
            )

        unsigned = dict(provenance)
        unsigned.pop("provenance_payload_sha256")

        if _canonical_sha256(unsigned) != claimed_hash:
            raise EvaluationInputError(
                "Provenance artifact payload hash mismatch."
            )

        if provenance.get("artifact_type") != "qdrant_index_build_provenance":
            raise EvaluationInputError(
                "Unexpected provenance artifact type."
            )

        if provenance.get("reconstruction_status") != "retrospective_verified":
            raise EvaluationInputError(
                "Provenance artifact is not retrospectively verified."
            )

        # 2. Validate input artifact hashes.
        inputs = provenance["input_artifacts"]

        manifest_hash = _sha256_bytes(MANIFEST_PATH.read_bytes())
        benchmark_hash = _sha256_bytes(BENCHMARK_PATH.read_bytes())

        if inputs["manifest_sha256"] != manifest_hash:
            raise EvaluationInputError(
                "Manifest SHA-256 does not match provenance."
            )

        if inputs["benchmark_sha256"] != benchmark_hash:
            raise EvaluationInputError(
                "Benchmark SHA-256 does not match provenance."
            )

        # 3. Validate logical corpus identity.
        logical = provenance["logical_corpus"]

        if logical["corpus_id"] != benchmark.get("corpus_id"):
            raise EvaluationInputError(
                "Benchmark corpus_id does not match provenance."
            )

        if logical["benchmark_version"] != benchmark.get("benchmark_version"):
            raise EvaluationInputError(
                "Benchmark version does not match provenance."
            )

        if logical["manifest_version"] != manifest.get("manifest_version"):
            raise EvaluationInputError(
                "Manifest version does not match provenance."
            )

        manifest_ids = {
            item["document_id"] for item in manifest["sources"]
        }

        benchmark_ids = {
            document_id
            for case in benchmark["answerable_cases"]
            for document_id in case["expected_document_ids"]
        }

        documents = provenance["documents"]

        if not isinstance(documents, list) or len(documents) != 7:
            raise EvaluationInputError(
                "Provenance must contain exactly seven documents."
            )

        document_ids = {
            item["document_id"] for item in documents
        }

        if len(document_ids) != len(documents):
            raise EvaluationInputError(
                "Duplicate document IDs in provenance."
            )

        if document_ids != manifest_ids or document_ids != benchmark_ids:
            raise EvaluationInputError(
                "Provenance document IDs do not match benchmark and manifest."
            )

        if set(logical["document_ids"]) != document_ids:
            raise EvaluationInputError(
                "Logical corpus document IDs do not match provenance documents."
            )

        # 4. Validate source snapshots.
        source_info = provenance["source_snapshot"]
        source_dir = Path(source_info["source_directory"])

        if not source_dir.is_absolute():
            source_dir = ROOT / source_dir

        if not source_dir.is_dir():
            raise EvaluationInputError(
                f"Provenance source directory does not exist: {source_dir}"
            )

        source_files = sorted(source_dir.glob("*.json"))

        if len(source_files) != 7:
            raise EvaluationInputError(
                f"Expected seven source JSON files; found {len(source_files)}."
            )

        source_data_by_id: dict[str, dict[str, Any]] = {}

        for path in source_files:
            source_data = _load_json(path)
            document_id = source_data.get("document_id")

            if not isinstance(document_id, str) or document_id in source_data_by_id:
                raise EvaluationInputError(
                    f"Invalid or duplicate source document ID: {document_id!r}."
                )

            source_data_by_id[document_id] = source_data

        if set(source_data_by_id) != document_ids:
            raise EvaluationInputError(
                "Source snapshot document IDs do not match provenance."
            )

        manifest_by_id = {
            item["document_id"]: item
            for item in manifest["sources"]
        }

        indexing = provenance["indexing_configuration"]

        if indexing["embedding_model_name"] != DEFAULT_EMBEDDING_MODEL:
            raise EvaluationInputError(
                "Provenance embedding model does not match evaluator."
            )

        if indexing["normalizer_version"] != NORMALIZER_VERSION:
            raise EvaluationInputError(
                "Provenance normalizer version does not match current code."
            )

        if indexing["chunker_version"] != CHUNKER_VERSION:
            raise EvaluationInputError(
                "Provenance chunker version does not match current code."
            )

        if (
            indexing["pipeline_fingerprint_version"]
            != PIPELINE_FINGERPRINT_VERSION
        ):
            raise EvaluationInputError(
                "Pipeline fingerprint version mismatch."
            )

        max_chars = indexing["max_chars"]
        overlap_chars = indexing["overlap_chars"]

        snapshot_rows = []
        normalized_contents: dict[str, str] = {}
        for item in documents:
            document_id = item["document_id"]
            source = source_data_by_id[document_id]
            manifest_source = manifest_by_id[document_id]

            content = source.get("content")

            if not isinstance(content, str):
                raise EvaluationInputError(
                    f"Source content is invalid for {document_id}."
                )

            raw_hash = _sha256_bytes(content.encode("utf-8"))
            normalized_content = normalize_text(content)
            normalized_contents[document_id] = normalized_content
            normalized_hash = _sha256_bytes(
                normalized_content.encode("utf-8")
            )

            if raw_hash != item["raw_content_sha256"]:
                raise EvaluationInputError(
                    f"Raw content hash mismatch for {document_id}."
                )

            if normalized_hash != item["normalized_content_sha256"]:
                raise EvaluationInputError(
                    f"Normalized content hash mismatch for {document_id}."
                )

            # Validate metadata against both source and manifest.
            metadata_pairs = (
                ("title", "title", "title"),
                ("source_url", "source_url", "url"),
                ("medical_domain", "medical_domain", "medical_domain"),
                ("source_last_updated", "source_last_updated", "source_last_updated"),
            )

            for sidecar_key, source_key, manifest_key in metadata_pairs:
                expected = item[sidecar_key]

                if source.get(source_key) != expected:
                    raise EvaluationInputError(
                        f"Source metadata mismatch: {document_id}/{source_key}."
                    )

                if manifest_source.get(manifest_key) != expected:
                    raise EvaluationInputError(
                        f"Manifest metadata mismatch: {document_id}/{manifest_key}."
                    )

            if item["normalizer_version"] != NORMALIZER_VERSION:
                raise EvaluationInputError(
                    f"Normalizer mismatch for {document_id}."
                )

            if item["chunker_version"] != CHUNKER_VERSION:
                raise EvaluationInputError(
                    f"Chunker mismatch for {document_id}."
                )

            if item["max_chars"] != max_chars or item["overlap_chars"] != overlap_chars:
                raise EvaluationInputError(
                    f"Chunking configuration mismatch for {document_id}."
                )

            chunk_ids = item["chunk_ids"]

            if (
                not isinstance(chunk_ids, list)
                or len(chunk_ids) != item["chunk_count"]
                or len(set(chunk_ids)) != len(chunk_ids)
                or any(not isinstance(value, str) or not value for value in chunk_ids)
            ):
                raise EvaluationInputError(
                    f"Invalid chunk IDs for {document_id}."
                )

            if item["fingerprint_validation"] != "pass":
                raise EvaluationInputError(
                    f"Fingerprint was not validated for {document_id}."
                )

            if item["embedding_model_name"] != DEFAULT_EMBEDDING_MODEL:
                raise EvaluationInputError(
                    f"Embedding model mismatch for {document_id}."
                )

            if item["embedding_dimensions"] != VECTOR_SIZE:
                raise EvaluationInputError(
                    f"Embedding dimensions mismatch for {document_id}."
                )

            # Revalidate fingerprint with the original Pydantic schema.
            try:
                IngestionRecord(
                    document_id=document_id,
                    raw_content_sha256=item["raw_content_sha256"],
                    normalized_content_sha256=item["normalized_content_sha256"],
                    normalizer_version=item["normalizer_version"],
                    chunker_version=item["chunker_version"],
                    max_chars=item["max_chars"],
                    overlap_chars=item["overlap_chars"],
                    chunk_count=item["chunk_count"],
                    pipeline_fingerprint=item["pipeline_fingerprint"],
                )
            except Exception as exc:
                raise EvaluationInputError(
                    f"Invalid pipeline fingerprint for {document_id}."
                ) from exc

            snapshot_rows.append(
                (document_id, item["raw_content_sha256"])
            )

        # 5. Recalculate source snapshot identity.
        snapshot_text = "".join(
            f"{document_id}:{content_hash}\n"
            for document_id, content_hash in sorted(snapshot_rows)
        )

        actual_snapshot_hash = _sha256_bytes(
            snapshot_text.encode("utf-8")
        )

        if actual_snapshot_hash != source_info["source_snapshot_sha256"]:
            raise EvaluationInputError(
                "Source snapshot SHA-256 mismatch."
            )

        # 6. Verify reconstructed index identity.
        identity_rows = sorted([
            [
                item["document_id"],
                item["normalized_content_sha256"],
                item["pipeline_fingerprint"],
            ]
            for item in documents
        ])

        collection = provenance["collection"]

        expected_identity = _canonical_sha256({
            "collection": collection["name"],
            "source_snapshot_sha256": actual_snapshot_hash,
            "vector_size": collection["vector_size"],
            "distance": collection["distance"],
            "embedding_model_name": indexing["embedding_model_name"],
            "documents": identity_rows,
        })

        if expected_identity != provenance["reconstructed_index_identity"]["sha256"]:
            raise EvaluationInputError(
                "Reconstructed index identity mismatch."
            )

        # 7. Validate collection configuration.
        if collection["name"] != collection_name:
            raise EvaluationInputError(
                "Requested collection does not match provenance collection."
            )

        if collection["vector_size"] != VECTOR_SIZE:
            raise EvaluationInputError(
                "Provenance vector size does not match evaluator."
            )

        if collection["distance"] != "Cosine":
            raise EvaluationInputError(
                "Unsupported vector distance in provenance."
            )

        if collection["active_chunk_points"] != sum(
            item["chunk_count"] for item in documents
        ):
            raise EvaluationInputError(
                "Active chunk total does not match provenance documents."
            )

    except EvaluationInputError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise EvaluationInputError(
            f"Invalid provenance artifact: {type(exc).__name__}: {exc}"
        ) from exc
    return normalized_contents

def _validate_live_qdrant_collection(
    provenance: dict[str, Any],
    normalized_contents: dict[str, str],
) -> dict[str, Any]:
    """Read-only audit of live Qdrant points against provenance and source."""

    collection_name = provenance["collection"]["name"]
    documents = provenance["documents"]

    document_by_id = {
        item["document_id"]: item
        for item in documents
    }

    expected_total = provenance["collection"]["total_points"]
    expected_chunks = provenance["collection"]["active_chunk_points"]

    client = QdrantClient(url=QDRANT_URL, timeout=30.0)

    try:
        if not client.collection_exists(collection_name):
            raise EvaluationInputError(
                f"Live Qdrant collection does not exist: {collection_name}"
            )

        # Read all points. This function must never mutate Qdrant.
        points = []
        offset = None

        while True:
            batch, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points.extend(batch)

            if next_offset is None:
                break

            offset = next_offset

        if len(points) != expected_total:
            raise EvaluationInputError(
                f"Live point count mismatch: "
                f"expected={expected_total}, actual={len(points)}"
            )

        chunks_by_document: dict[str, list[dict[str, Any]]] = {}
        states_by_document: dict[str, dict[str, Any]] = {}
        seen_point_ids = set()

        for point in points:
            if point.id in seen_point_ids:
                raise EvaluationInputError(
                    f"Duplicate point ID: {point.id}"
                )

            seen_point_ids.add(point.id)

            payload = point.payload

            if not isinstance(payload, dict):
                raise EvaluationInputError(
                    f"Point {point.id} has invalid payload."
                )

            record_type = payload.get("record_type")
            document_id = payload.get("document_id")

            if document_id not in document_by_id:
                raise EvaluationInputError(
                    f"Unexpected document_id in Qdrant: {document_id}"
                )

            if record_type == "document_state":
                if document_id in states_by_document:
                    raise EvaluationInputError(
                        f"Duplicate document_state: {document_id}"
                    )

                states_by_document[document_id] = payload

            elif record_type == "chunk":
                chunks_by_document.setdefault(
                    document_id, []
                ).append(payload)

            else:
                raise EvaluationInputError(
                    f"Unknown record_type in Qdrant: {record_type!r}"
                )

        # Validate document-state records.
        if set(states_by_document) != set(document_by_id):
            raise EvaluationInputError(
                "Document-state set does not match provenance documents."
            )

        for document_id, document in document_by_id.items():
            state = states_by_document[document_id]

            if (
                state.get("active_pipeline_fingerprint")
                != document["pipeline_fingerprint"]
            ):
                raise EvaluationInputError(
                    f"Active fingerprint mismatch: {document_id}"
                )

            if (
                state.get("embedding_model")
                != document["embedding_model_name"]
            ):
                raise EvaluationInputError(
                    f"Document-state embedding model mismatch: {document_id}"
                )

        # Validate chunk records.
        if sum(len(items) for items in chunks_by_document.values()) != expected_chunks:
            raise EvaluationInputError(
                "Live chunk count does not match provenance."
            )

        for document_id, document in document_by_id.items():
            chunks = chunks_by_document.get(document_id, [])

            if len(chunks) != document["chunk_count"]:
                raise EvaluationInputError(
                    f"Chunk count mismatch: {document_id}"
                )

            chunks.sort(key=lambda item: item.get("chunk_index", -1))

            expected_chunk_ids = document["chunk_ids"]

            actual_indices = [
                item.get("chunk_index")
                for item in chunks
            ]

            if actual_indices != list(range(document["chunk_count"])):
                raise EvaluationInputError(
                    f"Non-contiguous chunk indices: {document_id}"
                )

            actual_chunk_ids = [
                item.get("chunk_id")
                for item in chunks
            ]

            if actual_chunk_ids != expected_chunk_ids:
                raise EvaluationInputError(
                    f"Chunk ID/order mismatch: {document_id}"
                )

            normalized_source = normalized_contents[document_id]

            for chunk in chunks:
                # Validate identity and pipeline metadata.
                if (
                    chunk.get("pipeline_fingerprint")
                    != document["pipeline_fingerprint"]
                ):
                    raise EvaluationInputError(
                        f"Chunk fingerprint mismatch: {document_id}"
                    )

                if chunk.get("embedding_model") != document["embedding_model_name"]:
                    raise EvaluationInputError(
                        f"Chunk embedding model mismatch: {document_id}"
                    )

                if chunk.get("embedding_dimensions") != document["embedding_dimensions"]:
                    raise EvaluationInputError(
                        f"Chunk embedding dimensions mismatch: {document_id}"
                    )

                if chunk.get("normalized_content_sha256") != document["normalized_content_sha256"]:
                    raise EvaluationInputError(
                        f"Chunk normalized hash mismatch: {document_id}"
                    )

                if chunk.get("raw_content_sha256") != document["raw_content_sha256"]:
                    raise EvaluationInputError(
                        f"Chunk raw hash mismatch: {document_id}"
                    )

                if chunk.get("normalizer_version") != document["normalizer_version"]:
                    raise EvaluationInputError(
                        f"Chunk normalizer mismatch: {document_id}"
                    )

                if chunk.get("chunker_version") != document["chunker_version"]:
                    raise EvaluationInputError(
                        f"Chunker version mismatch: {document_id}"
                    )

                # Validate source offsets and exact chunk content.
                start = chunk.get("start_char")
                end = chunk.get("end_char")
                content = chunk.get("content")

                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or not isinstance(content, str)
                    or start < 0
                    or end < start
                    or end > len(normalized_source)
                ):
                    raise EvaluationInputError(
                        f"Invalid chunk offsets/content: {document_id}"
                    )

                if normalized_source[start:end] != content:
                    raise EvaluationInputError(
                        f"Chunk content does not match source offsets: "
                        f"{document_id}/{chunk['chunk_id']}"
                    )

                # Validate document metadata.
                for field in (
                    "title",
                    "source_url",
                    "medical_domain",
                    "source_last_updated",
                ):
                    if chunk.get(field) != document[field]:
                        raise EvaluationInputError(
                            f"Live payload metadata mismatch: "
                            f"{document_id}/{field}"
                        )

    except EvaluationInputError:
        raise
    except Exception as exc:
        raise EvaluationInputError(
            f"Live Qdrant validation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        client.close()

    return {
        "status": "pass",
        "collection": collection_name,
        "total_points_checked": len(points),
        "chunk_points_checked": expected_chunks,
        "document_state_points_checked": len(states_by_document),
        "documents_checked": len(document_by_id),
        "read_only": True,
    }

def _validate_benchmark_and_manifest(
    benchmark: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answerable = benchmark.get("answerable_cases")
    unanswerable = benchmark.get("unanswerable_from_this_pilot")
    source_rows = manifest.get("sources")
    if not isinstance(answerable, list) or not isinstance(unanswerable, list):
        raise EvaluationInputError("Benchmark must contain answerable_cases and unanswerable_from_this_pilot lists.")
    if not isinstance(source_rows, list) or not source_rows:
        raise EvaluationInputError("Manifest must contain a non-empty sources list.")

    sources: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if not isinstance(source, dict):
            raise EvaluationInputError("Manifest sources must be JSON objects.")
        document_id = source.get("document_id")
        domain = source.get("medical_domain")
        if not isinstance(document_id, str) or not isinstance(domain, str):
            raise EvaluationInputError("Every manifest source requires document_id and medical_domain.")
        if document_id in sources:
            raise EvaluationInputError(f"Duplicate document_id in manifest: {document_id}.")
        sources[document_id] = source

    query_ids: set[str] = set()
    for case in answerable:
        if not isinstance(case, dict):
            raise EvaluationInputError("Answerable benchmark cases must be JSON objects.")
        query_id = case.get("query_id")
        question = case.get("question")
        domain = case.get("medical_domain")
        expected_ids = case.get("expected_document_ids")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in query_ids:
            raise EvaluationInputError(f"Invalid or duplicate query_id: {query_id!r}.")
        query_ids.add(query_id)
        if not isinstance(question, str) or not question.strip():
            raise EvaluationInputError(f"Question {query_id} is blank.")
        if not isinstance(domain, str) or not domain.strip():
            raise EvaluationInputError(f"Question {query_id} has no medical_domain.")
        if not isinstance(expected_ids, list) or not expected_ids:
            raise EvaluationInputError(f"Question {query_id} must define expected_document_ids.")
        for document_id in expected_ids:
            if document_id not in sources:
                raise EvaluationInputError(
                    f"Question {query_id} references document_id not present in manifest: {document_id}."
                )
            if sources[document_id].get("medical_domain") != domain:
                raise EvaluationInputError(
                    f"Question {query_id} domain does not match expected source {document_id}."
                )

    for case in unanswerable:
        if not isinstance(case, dict):
            raise EvaluationInputError("Unanswerable benchmark cases must be JSON objects.")
        query_id = case.get("query_id")
        question = case.get("question")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in query_ids:
            raise EvaluationInputError(f"Invalid or duplicate query_id: {query_id!r}.")
        query_ids.add(query_id)
        if not isinstance(question, str) or not question.strip():
            raise EvaluationInputError(f"Unanswerable question {query_id} is blank.")

    if len(answerable) != 28 or len(unanswerable) != 7:
        raise EvaluationInputError(
            f"Expected 28 answerable and 7 out-of-scope questions; got {len(answerable)} and {len(unanswerable)}."
        )
    return answerable, unanswerable


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "chunk_id": result.chunk_id,
        "medical_domain": result.medical_domain,
        "title": result.title,
        "source_url": result.source_url,
        "score": float(result.score),
    }


def _metrics_dict(metrics: Any, *, top_k: int) -> dict[str, Any]:
    return {
        "expected_rank": metrics.expected_rank,
        "hit_at_1": metrics.hit_at_1,
        "hit_at_3": metrics.hit_at_3,
        f"hit_at_{top_k}": metrics.hit_at_k,
        f"reciprocal_rank_at_{top_k}": metrics.reciprocal_rank_at_k,
        "top1_domain_match": metrics.top1_domain_match,
        f"domain_filter_leakage_in_top_{top_k}": metrics.domain_filter_leakage,
    }


def _print_summary(label: str, summary: dict[str, Any], *, top_k: int) -> None:
    print(f"\n=== {label} ===")
    print(f"Queries: {summary['query_count']}")
    print(f"Hit@1: {summary['hit_at_1']}/{summary['query_count']} ({summary['hit_rate_at_1']:.1%})")
    print(f"Hit@3: {summary['hit_at_3']}/{summary['query_count']} ({summary['hit_rate_at_3']:.1%})")
    print(f"Hit@{top_k}: {summary['hit_at_k']}/{summary['query_count']} ({summary['hit_rate_at_k']:.1%})")
    print(f"MRR@{top_k}: {summary['mrr_at_k']:.4f}")
    if summary["top1_domain_accuracy"] is not None:
        print(f"Top-1 domain accuracy: {summary['top1_domain_accuracy']:.1%}")
    print(f"Domain-filter leakage in top 5: {summary['domain_filter_leakage']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 21))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--collection-name",
        type=str,
        default=COLLECTION_NAME,
        help="Qdrant collection to evaluate.",
    )
    parser.add_argument(
    "--provenance",
    type=Path,
    required=True,
    help="Retrospective provenance sidecar for the selected collection.",
    )
    args = parser.parse_args()
    provenance_path = args.provenance

    if not provenance_path.is_absolute():
        provenance_path = ROOT / provenance_path
    collection_name = args.collection_name.strip()
    if not collection_name:
        parser.error("--collection-name cannot be empty")

    print("=== MULTI-DOMAIN DENSE RETRIEVAL EVALUATION ===")
    print("BENCHMARK:", BENCHMARK_PATH)
    print("MANIFEST:", MANIFEST_PATH)
    print("COLLECTION:", collection_name)
    print("EMBEDDING MODEL:", DEFAULT_EMBEDDING_MODEL)
    print("TOP-K:", args.top_k)

    try:
        benchmark = _load_json(BENCHMARK_PATH)
        manifest = _load_json(MANIFEST_PATH)
        provenance = _load_json(provenance_path)

        answerable, unanswerable = _validate_benchmark_and_manifest(
            benchmark, manifest
        )

        normalized_contents = _validate_provenance(
            provenance=provenance,
            benchmark=benchmark,
            manifest=manifest,
            collection_name=args.collection_name,
        )

        live_qdrant_validation = _validate_live_qdrant_collection(
            provenance=provenance,
            normalized_contents=normalized_contents,
        )

        print("PROVENANCE VALIDATION: PASS")
        print("LIVE QDRANT VALIDATION: PASS")
        print(
            "LIVE POINTS CHECKED:",
            live_qdrant_validation["total_points_checked"],
        )

        print("PROVENANCE VALIDATION: PASS")
        print("PROVENANCE SIDECAR:", provenance_path)
        print(
            "SOURCE SNAPSHOT:",
            provenance["source_snapshot"]["source_snapshot_sha256"],
        )
        print(
            "INDEX IDENTITY:",
            provenance["reconstructed_index_identity"]["sha256"],
        )
    except EvaluationInputError as exc:
        print("EVALUATION INPUT FAILED:", exc, file=sys.stderr)
        return 1
    
    print(
        f"BENCHMARK VALIDATION: PASS ({len(answerable)} answerable, "
        f"{len(unanswerable)} out-of-scope, {len(manifest['sources'])} manifest sources)"
    )

    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=VECTOR_SIZE,
        distance="cosine",
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )

    answerable_unfiltered_metrics = []
    answerable_filtered_metrics = []
    per_domain: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"unfiltered": [], "domain_filtered": []}
    )
    answerable_results: list[dict[str, Any]] = []
    unanswerable_results: list[dict[str, Any]] = []

    try:
        with OllamaEmbeddingProvider(model=DEFAULT_EMBEDDING_MODEL, timeout=180.0) as embedder:
            with QdrantVectorStore(url=QDRANT_URL, config=config, timeout=30.0) as store:
                store.connect_existing_collection()

                # Preflight: all manifest documents must have active indexed chunks.
                for source in manifest["sources"]:
                    count = store.count(VectorSearchFilter(document_ids=(source["document_id"],)))
                    if count < 1:
                        raise EvaluationInputError(
                            f"No active indexed chunks for manifest document {source['document_id']}."
                        )
                for domain in sorted({source["medical_domain"] for source in manifest["sources"]}):
                    count = store.count(VectorSearchFilter(medical_domains=(domain,)))
                    if count < 1:
                        raise EvaluationInputError(f"No active chunks indexed for domain {domain}.")

                retriever = DenseRetriever(embedding_provider=embedder, vector_store=store)
                print("INDEX PREFLIGHT: PASS")

                for index, case in enumerate(answerable, start=1):
                    query = case["question"]
                    domain = case["medical_domain"]
                    expected_ids = case["expected_document_ids"]

                    unfiltered = retriever.retrieve(query, top_k=args.top_k)
                    filtered = retriever.retrieve(
                        query,
                        top_k=args.top_k,
                        filters=VectorSearchFilter(medical_domains=(domain,)),
                    )

                    unfiltered_metric = evaluate_ranked_results(
                        unfiltered,
                        expected_document_ids=expected_ids,
                        expected_domain=domain,
                        k=args.top_k,
                    )
                    filtered_metric = evaluate_ranked_results(
                        filtered,
                        expected_document_ids=expected_ids,
                        expected_domain=domain,
                        k=args.top_k,
                        require_domain_match=True,
                    )
                    answerable_unfiltered_metrics.append(unfiltered_metric)
                    answerable_filtered_metrics.append(filtered_metric)
                    per_domain[domain]["unfiltered"].append(unfiltered_metric)
                    per_domain[domain]["domain_filtered"].append(filtered_metric)

                    answerable_results.append(
                        {
                            "query_id": case["query_id"],
                            "medical_domain": domain,
                            "question": query,
                            "expected_document_ids": expected_ids,
                            "unfiltered": {
                                "metrics": _metrics_dict(unfiltered_metric, top_k=args.top_k),
                                "results": [_result_payload(item) for item in unfiltered],
                            },
                            "domain_filtered": {
                                "metrics": _metrics_dict(filtered_metric, top_k=args.top_k),
                                "results": [_result_payload(item) for item in filtered],
                            },
                        }
                    )
                    print(
                        f"[{index:02d}/{len(answerable)}] {case['query_id']} | {domain} | "
                        f"unfiltered_rank={unfiltered_metric.expected_rank} | "
                        f"domain_filtered_rank={filtered_metric.expected_rank}"
                    )

                for index, case in enumerate(unanswerable, start=1):
                    results = retriever.retrieve(case["question"], top_k=args.top_k)
                    unanswerable_results.append(
                        {
                            "query_id": case["query_id"],
                            "question": case["question"],
                            "reason": case.get("reason"),
                            "top_results": [_result_payload(item) for item in results],
                            "interpretation": (
                                "Diagnostic nearest-neighbor output only. No score threshold is used "
                                "to decide answerability or evidence sufficiency."
                            ),
                        }
                    )
                    top = results[0] if results else None
                    print(
                        f"[OOS {index:02d}/{len(unanswerable)}] {case['query_id']} | "
                        f"top_document={top.document_id if top else None} | "
                        f"top_score={f'{top.score:.4f}' if top else None}"
                    )

    except EvaluationInputError as exc:
        print("EVALUATION FAILED:", exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print("EVALUATION RUNTIME FAILED:", f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    unfiltered_summary = aggregate_metrics(answerable_unfiltered_metrics)
    filtered_summary = aggregate_metrics(answerable_filtered_metrics)
    per_domain_summary: dict[str, Any] = {}
    for domain, groups in sorted(per_domain.items()):
        per_domain_summary[domain] = {
            "unfiltered": aggregate_metrics(groups["unfiltered"]),
            "domain_filtered": aggregate_metrics(groups["domain_filtered"]),
        }

    _print_summary("ANSWERABLE – UNFILTERED RETRIEVAL", unfiltered_summary, top_k=args.top_k)
    _print_summary("ANSWERABLE – KNOWN-DOMAIN FILTERED RETRIEVAL", filtered_summary, top_k=args.top_k)
    print("\n=== PER-DOMAIN UNFILTERED HIT@5 ===")
    for domain, summary in per_domain_summary.items():
        m = summary["unfiltered"]
        print(f"{domain}: {m['hit_at_k']}/{m['query_count']} ({m['hit_rate_at_k']:.1%}), MRR@{args.top_k}={m['mrr_at_k']:.4f}")

    report_path = args.output
    if report_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = ROOT / "data" / "evaluation" / "results" / f"multidomain_retrieval_{timestamp}.json"
    elif not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "benchmark_version": benchmark.get("benchmark_version"),
        "corpus_id": benchmark.get("corpus_id"),
        "collection": collection_name,
        "provenance_validation": {
            "status": "pass",
            "artifact_type": provenance["artifact_type"],
            "reconstruction_status": provenance["reconstruction_status"],
            "sidecar_path": str(provenance_path),
            "sidecar_file_sha256": _sha256_bytes(
                provenance_path.read_bytes()
            ),
            "sidecar_payload_sha256": provenance["provenance_payload_sha256"],
            "source_snapshot_sha256": (
                provenance["source_snapshot"]["source_snapshot_sha256"]
            ),
            "reconstructed_index_identity_sha256": (
                provenance["reconstructed_index_identity"]["sha256"]
            ),
            "historical_git_commit_status": (
                provenance["code_provenance"]["historical_index_build_commit_status"]
            ),
            "historical_embedding_digest_status": (
                provenance["embedding_artifact"]["historical_model_digest_status"]
            ),
        },
        "live_qdrant_validation": live_qdrant_validation,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE,
        "top_k": args.top_k,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "answerable": {
            "query_count": len(answerable_results),
            "unfiltered_metrics": unfiltered_summary,
            "known_domain_filtered_metrics": filtered_summary,
            "per_domain": per_domain_summary,
            "cases": answerable_results,
        },
        "out_of_scope_diagnostics": {
            "query_count": len(unanswerable_results),
            "warning": (
                "These seven queries are outside the declared pilot scope. Their nearest-neighbor "
                "scores are diagnostic only and cannot establish answerability or abstention."
            ),
            "cases": unanswerable_results,
        },
        "interpretation_limits": [
            "Hit@k measures whether an expected document appears in the top-k; it does not prove that the retrieved chunk contains every expected concept.",
            "MRR@k measures the rank of the first expected document; it does not measure clinical correctness.",
            "Similarity scores are not probabilities, calibrated confidence, or evidence-sufficiency thresholds.",
            "Domain-filtered results use the expected domain as an oracle filter and are diagnostic, not a measure of automatic domain classification.",
            "The benchmark contains 28 authored answerable questions and seven out-of-scope diagnostics over only seven source summaries; it is a pilot, not a validated clinical benchmark.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nEVALUATION STATUS: PASS")
    print("REPORT:", report_path)
    print("NOTE: Retrieval ranking evaluated; Qwen answer quality and abstention are not evaluated in this phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
