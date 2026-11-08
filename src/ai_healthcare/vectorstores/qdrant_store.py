"""Qdrant Server adapter with recoverable document-version activation."""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from ai_healthcare.schemas.vector import (
    VectorRecord,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStoreConfig,
)
from ai_healthcare.vectorstores.base import (
    CollectionConfigurationError,
    VectorDimensionError,
    VectorStore,
    VectorStoreError,
)


_RECORD_TYPE_FIELD = "record_type"
_RECORD_TYPE_CHUNK = "chunk"
_RECORD_TYPE_DOCUMENT_STATE = "document_state"
_PAYLOAD_INDEX_FIELDS = (
    _RECORD_TYPE_FIELD,
    "document_id",
    "chunk_id",
    "source_type",
    "language",
    "pipeline_fingerprint",
)


class QdrantVectorStore(VectorStore):
    """Synchronous Qdrant Server adapter using the REST API by default.

    A deterministic document-state point is the authoritative pointer to the
    active pipeline fingerprint for each document. New chunk points are staged
    first, then the state pointer is switched, and only then are stale chunks
    deleted. Retrieval checks the state pointer and filters out stale versions,
    including when stale-point cleanup failed or the process restarted.

    The lock serializes mutations made through this instance. Multi-process
    writers still require an external transactional coordinator/lock.
    """

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        timeout: float = 10.0,
        config: VectorStoreConfig | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must not be blank.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be numeric.")
        if not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("timeout must be greater than zero and finite.")

        self.config = config or VectorStoreConfig()
        self._owns_client = client is None
        self._client = client or QdrantClient(
            url=url.rstrip("/"),
            api_key=api_key,
            timeout=float(timeout),
        )
        self._ready = False
        self._mutation_lock = threading.RLock()

    def ensure_collection(self) -> None:
        """Create and validate the configured single-vector collection."""
        self._ready = False
        try:
            if not self._client.collection_exists(self.config.collection_name):
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            info = self._client.get_collection(self.config.collection_name)
            vector_config = info.config.params.vectors

            # This adapter intentionally supports one unnamed dense vector only.
            if isinstance(vector_config, dict):
                raise CollectionConfigurationError(
                    "Configured collection uses named vectors; this adapter expects one unnamed dense vector."
                )
            if vector_config.size != self.config.vector_size:
                raise CollectionConfigurationError(
                    "Collection vector size does not match VectorStoreConfig."
                )
            if vector_config.distance != models.Distance.COSINE:
                raise CollectionConfigurationError(
                    "Collection distance must be Cosine."
                )

            payload_schema = info.payload_schema or {}
            for field_name in _PAYLOAD_INDEX_FIELDS:
                existing_index = payload_schema.get(field_name)
                if existing_index is not None:
                    existing_type = getattr(existing_index, "data_type", None)
                    if (
                        existing_type is not None
                        and existing_type != models.PayloadSchemaType.KEYWORD
                    ):
                        raise CollectionConfigurationError(
                            f"Payload index '{field_name}' must use KEYWORD schema."
                        )
                    continue

                self._client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

            self._ready = True
        except CollectionConfigurationError:
            raise
        except Exception as exc:
            self._ready = False
            raise VectorStoreError("Failed to initialize the Qdrant collection.") from exc

    def replace_document(self, records: Sequence[VectorRecord]) -> None:
        """Activate a complete new document version and clean up stale chunks.

        Sequence:
        1. Upsert the complete new chunk set (staged, not yet active).
        2. Upsert the document-state point to switch the active fingerprint.
        3. Delete chunks belonging to older fingerprints.

        If step 1 fails, the old version remains active. If step 2 fails, the
        old version remains active and the staged points are ignored by search.
        If step 3 fails, the new version remains active; stale points are
        ignored by search and a retry can complete cleanup.
        """
        self._require_ready()
        validated_records = self._validate_document_records(records)
        document_id = validated_records[0].document.document_id
        current_fingerprint = validated_records[0].pipeline_fingerprint

        points = [
            models.PointStruct(
                id=self._point_id(record),
                vector=list(record.vector),
                payload=self._build_payload(record),
            )
            for record in validated_records
        ]

        with self._mutation_lock:
            try:
                # Stage all new chunks before making this version retrievable.
                self._client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True,
                )

                # One deterministic point is the authoritative active-version pointer.
                state_point = models.PointStruct(
                    id=self._document_state_point_id(document_id),
                    vector=self._state_vector(),
                    payload={
                        _RECORD_TYPE_FIELD: _RECORD_TYPE_DOCUMENT_STATE,
                        "document_id": document_id,
                        "active_pipeline_fingerprint": current_fingerprint,
                        "embedding_model": self.config.embedding_model,
                    },
                )
                self._client.upsert(
                    collection_name=self.config.collection_name,
                    points=[state_point],
                    wait=True,
                )

                stale_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=_RECORD_TYPE_FIELD,
                            match=models.MatchValue(value=_RECORD_TYPE_CHUNK),
                        ),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="pipeline_fingerprint",
                            match=models.MatchValue(value=current_fingerprint),
                        )
                    ],
                )
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.FilterSelector(filter=stale_filter),
                    wait=True,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to replace indexed document '{document_id}'."
                ) from exc

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self._require_ready()
        vector = self._validate_query_vector(query_vector)

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100.")
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise TypeError("score_threshold must be numeric.")
            if not math.isfinite(float(score_threshold)):
                raise ValueError("score_threshold must be finite.")

        query_filter = self._build_chunk_filter(filters)
        page_size = min(100, max(16, limit * 2))
        offset = 0
        active_results: list[VectorSearchResult] = []

        try:
            while len(active_results) < limit:
                response = self._client.query_points(
                    collection_name=self.config.collection_name,
                    query=vector,
                    query_filter=query_filter,
                    limit=page_size,
                    offset=offset,
                    score_threshold=(
                        float(score_threshold) if score_threshold is not None else None
                    ),
                    with_payload=True,
                    with_vectors=False,
                )
                page = list(response.points)
                if not page:
                    break

                document_ids = {
                    str(hit.payload["document_id"])
                    for hit in page
                    if hit.payload and hit.payload.get("document_id")
                }
                active_versions = self._get_active_fingerprints(document_ids)

                for hit in page:
                    if not hit.payload:
                        continue
                    document_id = str(hit.payload.get("document_id", ""))
                    fingerprint = hit.payload.get("pipeline_fingerprint")
                    if not document_id or active_versions.get(document_id) != fingerprint:
                        continue
                    active_results.append(self._map_search_hit(hit))
                    if len(active_results) >= limit:
                        break

                offset += len(page)
                if len(page) < page_size:
                    break

            return tuple(active_results[:limit])
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant vector search failed.") from exc

    def delete_document(self, document_id: str) -> None:
        """Deactivate a document before deleting its chunk points.

        If chunk deletion fails, the orphaned chunks remain inaccessible because
        search requires a matching document-state point.
        """
        self._require_ready()
        document_id = self._validate_nonblank(document_id, "document_id")
        with self._mutation_lock:
            try:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.PointIdsList(
                        points=[self._document_state_point_id(document_id)]
                    ),
                    wait=True,
                )
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=_RECORD_TYPE_FIELD,
                                    match=models.MatchValue(value=_RECORD_TYPE_CHUNK),
                                ),
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                            ]
                        )
                    ),
                    wait=True,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to delete document '{document_id}'."
                ) from exc

    def delete_chunks(self, document_id: str, chunk_ids: Sequence[str]) -> None:
        self._require_ready()
        document_id = self._validate_nonblank(document_id, "document_id")
        validated_ids = self._validate_string_sequence(chunk_ids, "chunk_ids")
        if not validated_ids:
            return

        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=_RECORD_TYPE_FIELD,
                        match=models.MatchValue(value=_RECORD_TYPE_CHUNK),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=list(validated_ids)),
                    ),
                ]
            )
        )
        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=selector,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete selected chunks from '{document_id}'."
            ) from exc

    def count(self, filters: VectorSearchFilter | None = None) -> int:
        """Count retrievable (active-version) chunks, not orphaned/staged points."""
        self._require_ready()
        scroll_filter = self._build_chunk_filter(filters)
        offset: Any = None
        total = 0
        try:
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self.config.collection_name,
                    scroll_filter=scroll_filter,
                    limit=256,
                    offset=offset,
                    with_payload=["document_id", "pipeline_fingerprint"],
                    with_vectors=False,
                )
                if not records:
                    break

                document_ids = {
                    str(record.payload["document_id"])
                    for record in records
                    if record.payload and record.payload.get("document_id")
                }
                active_versions = self._get_active_fingerprints(document_ids)
                total += sum(
                    1
                    for record in records
                    if record.payload
                    and active_versions.get(str(record.payload.get("document_id", "")))
                    == record.payload.get("pipeline_fingerprint")
                )

                if next_offset is None:
                    break
                offset = next_offset
            return total
        except Exception as exc:
            raise VectorStoreError("Failed to count active Qdrant chunks.") from exc

    def close(self) -> None:
        """Close only clients created by this adapter."""
        if self._owns_client:
            self._client.close()
        self._ready = False

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _require_ready(self) -> None:
        if not self._ready:
            self.ensure_collection()

    def _get_active_fingerprints(self, document_ids: set[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        states = self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=[self._document_state_point_id(item) for item in sorted(document_ids)],
            with_payload=True,
            with_vectors=False,
        )
        result: dict[str, str] = {}
        for state in states:
            payload = state.payload or {}
            if payload.get(_RECORD_TYPE_FIELD) != _RECORD_TYPE_DOCUMENT_STATE:
                continue
            document_id = payload.get("document_id")
            fingerprint = payload.get("active_pipeline_fingerprint")
            if isinstance(document_id, str) and isinstance(fingerprint, str):
                result[document_id] = fingerprint
        return result

    def _validate_document_records(
        self,
        records: Sequence[VectorRecord],
    ) -> tuple[VectorRecord, ...]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of VectorRecord objects.")
        items = tuple(records)
        if not items:
            raise ValueError("At least one VectorRecord is required.")
        if not all(isinstance(item, VectorRecord) for item in items):
            raise TypeError("Every item must be a VectorRecord.")

        first = items[0]
        document_id = first.document.document_id
        fingerprint = first.pipeline_fingerprint
        embedding_model = first.embedding_model
        metadata = first.document.model_dump(exclude={"content"})
        ingestion_metadata = first.ingestion.model_dump(exclude={"processed_at"})

        chunk_ids: set[str] = set()
        indices: list[int] = []
        for item in items:
            if item.document.document_id != document_id:
                raise ValueError("All records must belong to one document.")
            if item.pipeline_fingerprint != fingerprint:
                raise ValueError("All records must share one pipeline fingerprint.")
            if item.embedding_model != embedding_model:
                raise ValueError("All records must use the same embedding model.")
            if item.document.model_dump(exclude={"content"}) != metadata:
                raise ValueError("Document metadata must be consistent across records.")
            if item.ingestion.model_dump(exclude={"processed_at"}) != ingestion_metadata:
                raise ValueError("Ingestion metadata must be consistent across records.")
            if item.embedding_model != self.config.embedding_model:
                raise VectorStoreError(
                    "Record embedding model does not match VectorStoreConfig."
                )
            if len(item.vector) != self.config.vector_size:
                raise VectorDimensionError(
                    f"Expected {self.config.vector_size}-dimensional embeddings; "
                    f"received {len(item.vector)} dimensions."
                )
            if item.chunk.chunk_id in chunk_ids:
                raise ValueError("Duplicate chunk_id in document replacement.")
            chunk_ids.add(item.chunk.chunk_id)
            indices.append(item.chunk.chunk_index)

        if len(items) != first.ingestion.chunk_count:
            raise ValueError("The replacement must contain every chunk in the ingestion record.")
        if sorted(indices) != list(range(first.ingestion.chunk_count)):
            raise ValueError("Chunk indices must be complete and contiguous from zero.")
        return items

    def _validate_query_vector(self, query_vector: Sequence[float]) -> list[float]:
        if isinstance(query_vector, (str, bytes)) or not isinstance(query_vector, Sequence):
            raise TypeError("query_vector must be a sequence of numeric values.")
        if len(query_vector) != self.config.vector_size:
            raise VectorDimensionError(
                f"Expected a {self.config.vector_size}-dimensional query vector; "
                f"received {len(query_vector)} dimensions."
            )

        result: list[float] = []
        for value in query_vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("query_vector must contain only numeric values.")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("query_vector values must be finite.")
            result.append(number)
        return result

    @staticmethod
    def _validate_nonblank(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")
        if not value.strip():
            raise ValueError(f"{field_name} must not be blank.")
        return value

    @classmethod
    def _validate_string_sequence(
        cls,
        values: Sequence[str],
        field_name: str,
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{field_name} must be a sequence of strings.")
        result = tuple(cls._validate_nonblank(value, field_name) for value in values)
        if len(result) != len(set(result)):
            raise ValueError(f"{field_name} cannot contain duplicates.")
        return result

    @staticmethod
    def _build_filter(
        filters: VectorSearchFilter | None,
    ) -> models.Filter | None:
        if filters is None:
            return None
        if not isinstance(filters, VectorSearchFilter):
            raise TypeError("filters must be a VectorSearchFilter or None.")

        conditions: list[models.FieldCondition] = []
        for field_name, values in (
            ("document_id", filters.document_ids),
            ("source_type", filters.source_types),
            ("language", filters.languages),
        ):
            if not values:
                continue
            match = (
                models.MatchValue(value=values[0])
                if len(values) == 1
                else models.MatchAny(any=list(values))
            )
            conditions.append(models.FieldCondition(key=field_name, match=match))

        return models.Filter(must=conditions) if conditions else None

    @classmethod
    def _build_chunk_filter(
        cls,
        filters: VectorSearchFilter | None,
    ) -> models.Filter:
        base_filter = cls._build_filter(filters)
        conditions = [
            models.FieldCondition(
                key=_RECORD_TYPE_FIELD,
                match=models.MatchValue(value=_RECORD_TYPE_CHUNK),
            )
        ]
        if base_filter is not None and base_filter.must:
            conditions.extend(base_filter.must)
        return models.Filter(must=conditions)

    @staticmethod
    def _document_state_point_id(document_id: str) -> str:
        identity = f"ai-healthcare:document-state:{document_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    def _state_vector(self) -> list[float]:
        # Non-zero deterministic vector. State points are always excluded from search.
        return [1.0] + [0.0] * (self.config.vector_size - 1)

    @staticmethod
    def _point_id(record: VectorRecord) -> str:
        identity = (
            f"ai-healthcare:{record.document.document_id}:"
            f"{record.pipeline_fingerprint}:{record.chunk.chunk_id}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    @staticmethod
    def _build_payload(record: VectorRecord) -> dict[str, Any]:
        document = record.document
        chunk = record.chunk
        ingestion = record.ingestion
        return {
            _RECORD_TYPE_FIELD: _RECORD_TYPE_CHUNK,
            "document_id": document.document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "title": document.title,
            "source_type": document.source_type,
            "source_name": document.source_name,
            "source_url": str(document.source_url) if document.source_url else None,
            "doi": document.doi,
            "publication_date": (
                document.publication_date.isoformat()
                if document.publication_date else None
            ),
            "retrieved_at": document.retrieved_at.isoformat(),
            "language": document.language,
            "license": document.license,
            "section_title": chunk.section_title,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "raw_content_sha256": ingestion.raw_content_sha256,
            "normalized_content_sha256": ingestion.normalized_content_sha256,
            "normalizer_version": ingestion.normalizer_version,
            "chunker_version": ingestion.chunker_version,
            "pipeline_fingerprint": ingestion.pipeline_fingerprint,
            "embedding_model": record.embedding_model,
            "embedding_dimensions": len(record.vector),
        }

    @staticmethod
    def _map_search_hit(hit: Any) -> VectorSearchResult:
        if hit.payload is None:
            raise VectorStoreError("Qdrant returned a point without payload.")
        data = dict(hit.payload)
        data.pop(_RECORD_TYPE_FIELD, None)
        data["point_id"] = str(hit.id)
        data["score"] = float(hit.score)
        try:
            return VectorSearchResult.model_validate(data)
        except Exception as exc:
            raise VectorStoreError("Qdrant returned an invalid vector payload.") from exc
