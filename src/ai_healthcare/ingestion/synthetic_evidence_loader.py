import json
import re
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_healthcare.schemas.document import MedicalDocument

__all__ = ["SyntheticEvidenceError", "load_synthetic_evidence"]

_EVIDENCE_ID_PATTERN = re.compile(r"E[1-9][0-9]{0,2}")
_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_FORBIDDEN_DIRECTORY_NAMES = frozenset({"patients"})

# "doc_" plus the 124 characters MedicalDocument.document_id allows.
_MAX_DOCUMENT_ID_LENGTH = 128
_FIXTURE_LANGUAGE = "vi"


class SyntheticEvidenceError(ValueError):
    """Raised when a synthetic evidence fixture cannot be loaded safely.

    Messages never include evidence text.
    """


def _evidence_sort_key(evidence_id: str) -> int:
    return int(evidence_id[1:])


def _validate_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or pathlib.Path.")

    file_path = Path(path)

    if any(
        part.lower() in _FORBIDDEN_DIRECTORY_NAMES
        for part in file_path.parts
    ):
        raise SyntheticEvidenceError(
            "Files under a 'patients' directory cannot be loaded "
            "as medical evidence."
        )

    if file_path.suffix.lower() != ".json":
        raise SyntheticEvidenceError(
            "Evidence file must have a .json extension."
        )

    return file_path


def _validate_required_ids(
    required_ids: Collection[str] | None,
) -> tuple[str, ...]:
    if required_ids is None:
        return ()

    if isinstance(required_ids, (str, bytes)) or not isinstance(
        required_ids, Collection
    ):
        raise TypeError(
            "required_ids must be a collection of evidence id strings."
        )

    ids = tuple(required_ids)

    for evidence_id in ids:
        if not isinstance(
            evidence_id, str
        ) or not _EVIDENCE_ID_PATTERN.fullmatch(evidence_id):
            raise SyntheticEvidenceError(
                "required_ids must contain valid evidence ids "
                "such as 'E1'."
            )

    return ids


def _read_text(file_path: Path) -> str:
    try:
        # utf-8-sig tolerates a BOM written by some Windows tools.
        return file_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        raise SyntheticEvidenceError(
            "Evidence file could not be read as UTF-8 text."
        ) from None


def _parse_json_rejecting_duplicates(text: str) -> Any:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SyntheticEvidenceError(
                    "Evidence file contains duplicate keys."
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError:
        raise SyntheticEvidenceError(
            "Evidence file is not valid JSON."
        ) from None


def _validate_entries(
    data: Any,
    required_ids: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(data, dict):
        raise SyntheticEvidenceError(
            "Evidence file must contain a JSON object."
        )

    if not data:
        raise SyntheticEvidenceError(
            "Evidence file contains no evidence entries."
        )

    for key, value in data.items():
        if not _EVIDENCE_ID_PATTERN.fullmatch(key):
            raise SyntheticEvidenceError(
                "Evidence file contains a key that is not a valid "
                "evidence id (expected E1, E2, ...)."
            )

        if not isinstance(value, str) or not value.strip():
            raise SyntheticEvidenceError(
                f"Evidence {key} must be a non-empty string."
            )

    missing = sorted(
        set(required_ids) - data.keys(),
        key=_evidence_sort_key,
    )

    if missing:
        raise SyntheticEvidenceError(
            f"Missing required evidence ids: {', '.join(missing)}."
        )

    return data


def load_synthetic_evidence(
    path: str | Path,
    *,
    required_ids: Collection[str] | None = None,
    retrieved_at: datetime | None = None,
) -> tuple[MedicalDocument, ...]:
    """Load a synthetic evidence fixture as one document per evidence key.

    Args:
        path: JSON file shaped like ``{"E1": "...", "E2": "..."}``.
        required_ids: Evidence ids that must be present (e.g. ("E1", "E2")).
        retrieved_at: Timezone-aware timestamp shared by all documents.
            Defaults to the current UTC time, computed once per call.

    Returns:
        Documents ordered by numeric evidence id. ``content`` is the
        original string, unmodified.

    Raises:
        TypeError: for arguments of the wrong type.
        SyntheticEvidenceError: for any unsafe or invalid fixture.
    """
    file_path = _validate_path(path)
    required = _validate_required_ids(required_ids)

    if retrieved_at is not None:
        if not isinstance(retrieved_at, datetime):
            raise TypeError("retrieved_at must be a datetime.")
        if retrieved_at.utcoffset() is None:
            raise SyntheticEvidenceError(
                "retrieved_at must be timezone-aware."
            )

    timestamp = (
        retrieved_at
        if retrieved_at is not None
        else datetime.now(timezone.utc)
    )

    data = _parse_json_rejecting_duplicates(_read_text(file_path))
    entries = _validate_entries(data, required)

    safe_stem = _UNSAFE_STEM_CHARS.sub("-", file_path.stem).strip("._-")
    if not safe_stem:
        raise SyntheticEvidenceError(
            "Evidence file name cannot be turned into a document id."
        )

    documents: list[MedicalDocument] = []

    for evidence_id in sorted(entries, key=_evidence_sort_key):
        document_id = f"doc_synthetic-{safe_stem}-{evidence_id}"

        if len(document_id) > _MAX_DOCUMENT_ID_LENGTH:
            raise SyntheticEvidenceError(
                "Evidence file name is too long for a document id."
            )

        try:
            documents.append(
                MedicalDocument(
                    document_id=document_id,
                    title=(
                        f"Synthetic evidence {evidence_id} "
                        f"from {file_path.name}"
                    ),
                    source_type="synthetic_fixture",
                    source_name=f"synthetic_fixture/{file_path.name}",
                    language=_FIXTURE_LANGUAGE,
                    retrieved_at=timestamp,
                    content=entries[evidence_id],
                )
            )
        except ValidationError:
            raise SyntheticEvidenceError(
                f"Evidence {evidence_id} failed MedicalDocument "
                "validation."
            ) from None

    return tuple(documents)