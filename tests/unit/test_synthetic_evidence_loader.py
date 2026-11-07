"""Loader for synthetic evidence fixtures (Step 3B.3.2).

Converts a fixture such as ``data/synthetic/evidence/latex_allergy.json``
into one ``MedicalDocument`` per evidence key (E1, E2, ...), so that the
evidence identity survives chunking, retrieval and citation.

Design note: "one evidence key = one MedicalDocument" is an adapter
decision for this synthetic fixture only. It is NOT a general rule for
real PDF/HTML documents, where several evidence fragments belong to one
source. Revisit with an EvidenceRecord / SourceFragment schema when real
documents are ingested.

This module never reads patient records and never invents provenance.
Error messages never contain evidence text.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_healthcare.ingestion.chunker import chunk_document
from ai_healthcare.ingestion.synthetic_evidence_loader import (
    SyntheticEvidenceError,
    load_synthetic_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_FILE = (
    ROOT / "data" / "synthetic" / "evidence" / "latex_allergy.json"
)
PATIENT_FILE = ROOT / "data" / "synthetic" / "patients" / "SYN-001.json"

FIXED_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
SECRET = "SECRET-MARKER-123"


def write_json(
    path: Path,
    payload: object,
    encoding: str = "utf-8",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding=encoding,
    )
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- Happy path -----------------------------------------------------------


def test_real_fixture_loads_as_two_documents():
    documents = load_synthetic_evidence(
        EVIDENCE_FILE, required_ids=("E1", "E2")
    )

    assert [d.document_id for d in documents] == [
        "doc_synthetic-latex_allergy-E1",
        "doc_synthetic-latex_allergy-E2",
    ]


def test_real_fixture_content_preserved_exactly():
    expected = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8-sig"))

    documents = load_synthetic_evidence(EVIDENCE_FILE)

    assert documents[0].content == expected["E1"]
    assert documents[1].content == expected["E2"]


def test_documents_carry_synthetic_metadata_without_fake_provenance():
    documents = load_synthetic_evidence(EVIDENCE_FILE)

    for evidence_id, document in zip(("E1", "E2"), documents):
        assert document.source_type == "synthetic_fixture"
        assert document.language == "vi"
        assert document.source_name == (
            "synthetic_fixture/latex_allergy.json"
        )
        assert evidence_id in document.title
        assert document.source_url is None
        assert document.doi is None
        assert document.publication_date is None
        assert document.license is None


def test_retrieved_at_can_be_injected_and_is_shared():
    injected = load_synthetic_evidence(
        EVIDENCE_FILE, retrieved_at=FIXED_TIME
    )
    assert {d.retrieved_at for d in injected} == {FIXED_TIME}

    default = load_synthetic_evidence(EVIDENCE_FILE)
    assert len({d.retrieved_at for d in default}) == 1
    assert default[0].retrieved_at.utcoffset() is not None


def test_document_ids_are_stable_across_loads():
    first = load_synthetic_evidence(EVIDENCE_FILE)
    second = load_synthetic_evidence(EVIDENCE_FILE)

    assert [d.document_id for d in first] == [
        d.document_id for d in second
    ]


def test_documents_are_ordered_by_numeric_evidence_id(tmp_path):
    path = write_json(
        tmp_path / "order.json",
        {"E10": "c", "E2": "b", "E1": "a"},
    )

    documents = load_synthetic_evidence(path)

    assert [d.document_id for d in documents] == [
        "doc_synthetic-order-E1",
        "doc_synthetic-order-E2",
        "doc_synthetic-order-E10",
    ]


def test_content_whitespace_is_not_normalized(tmp_path):
    original = "  keep  inner   spaces  "
    path = write_json(tmp_path / "ws.json", {"E1": original, "E2": "b"})

    documents = load_synthetic_evidence(path)

    assert documents[0].content == original


def test_utf8_bom_is_tolerated(tmp_path):
    path = write_json(
        tmp_path / "bom.json",
        {"E1": "a", "E2": "b"},
        encoding="utf-8-sig",
    )

    documents = load_synthetic_evidence(path)

    assert [d.content for d in documents] == ["a", "b"]


# --- Invalid structure ----------------------------------------------------


def test_missing_required_key_raises(tmp_path):
    path = write_json(tmp_path / "e.json", {"E1": "a"})

    with pytest.raises(SyntheticEvidenceError, match="E2"):
        load_synthetic_evidence(path, required_ids=("E1", "E2"))


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_empty_or_blank_evidence_raises(tmp_path, value):
    path = write_json(tmp_path / "e.json", {"E1": "text", "E2": value})

    with pytest.raises(SyntheticEvidenceError, match="E2"):
        load_synthetic_evidence(path)


@pytest.mark.parametrize("value", [123, None, ["x"], {"a": 1}])
def test_non_string_evidence_raises(tmp_path, value):
    path = write_json(tmp_path / "e.json", {"E1": "text", "E2": value})

    with pytest.raises(SyntheticEvidenceError, match="E2"):
        load_synthetic_evidence(path)


@pytest.mark.parametrize(
    "key", ["evidence1", "e1", "E0", "E01", "E1000"]
)
def test_invalid_evidence_key_raises(tmp_path, key):
    path = write_json(tmp_path / "e.json", {key: "text"})

    with pytest.raises(SyntheticEvidenceError, match="evidence id"):
        load_synthetic_evidence(path)


def test_empty_object_raises(tmp_path):
    path = write_json(tmp_path / "e.json", {})

    with pytest.raises(SyntheticEvidenceError, match="no evidence"):
        load_synthetic_evidence(path)


@pytest.mark.parametrize("payload", [[], "text", 5])
def test_non_object_top_level_raises(tmp_path, payload):
    path = write_json(tmp_path / "e.json", payload)

    with pytest.raises(SyntheticEvidenceError, match="JSON object"):
        load_synthetic_evidence(path)


def test_malformed_json_raises(tmp_path):
    path = write_text(tmp_path / "e.json", "{not json")

    with pytest.raises(SyntheticEvidenceError, match="not valid JSON"):
        load_synthetic_evidence(path)


def test_duplicate_keys_raise(tmp_path):
    path = write_text(
        tmp_path / "e.json", '{"E1": "a", "E1": "b"}'
    )

    with pytest.raises(SyntheticEvidenceError, match="duplicate"):
        load_synthetic_evidence(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(SyntheticEvidenceError, match="read"):
        load_synthetic_evidence(tmp_path / "missing.json")


def test_non_json_suffix_raises(tmp_path):
    path = write_json(tmp_path / "evidence.txt", {"E1": "a"})

    with pytest.raises(SyntheticEvidenceError, match=r"\.json"):
        load_synthetic_evidence(path)


# --- Patient data must never be loaded as evidence -----------------------


def test_patients_directory_is_rejected(tmp_path):
    with pytest.raises(SyntheticEvidenceError, match="patients"):
        load_synthetic_evidence(PATIENT_FILE)

    valid_payload_in_patients_dir = write_json(
        tmp_path / "patients" / "fixture.json", {"E1": "a"}
    )
    with pytest.raises(SyntheticEvidenceError, match="patients"):
        load_synthetic_evidence(valid_payload_in_patients_dir)


def test_patient_record_shaped_content_is_not_evidence(tmp_path):
    path = write_json(
        tmp_path / "evidence" / "fake.json",
        {
            "patient_id": "SYN-001",
            "display_name": "Synthetic Person",
            "allergy_name": "latex",
            "allergy_status": "NOT_ASSESSED",
        },
    )

    with pytest.raises(SyntheticEvidenceError, match="evidence id"):
        load_synthetic_evidence(path)


# --- Error hygiene --------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"E1": SECRET, "E2": "   "},
        {"E1": SECRET, "E2": [SECRET]},
        {"E1": SECRET, "note": SECRET},
    ],
)
def test_error_messages_do_not_leak_evidence_content(tmp_path, payload):
    path = write_json(tmp_path / "leak.json", payload)

    with pytest.raises(SyntheticEvidenceError) as exc_info:
        load_synthetic_evidence(path)

    assert SECRET not in str(exc_info.value)


# --- Argument validation --------------------------------------------------


def test_path_must_be_str_or_path():
    with pytest.raises(TypeError):
        load_synthetic_evidence(123)  # type: ignore[arg-type]


def test_required_ids_must_not_be_a_plain_string(tmp_path):
    with pytest.raises(TypeError):
        load_synthetic_evidence(
            tmp_path / "x.json",
            required_ids="E1",  # type: ignore[arg-type]
        )


def test_required_ids_must_be_valid_evidence_ids(tmp_path):
    with pytest.raises(SyntheticEvidenceError, match="required_ids"):
        load_synthetic_evidence(
            tmp_path / "x.json", required_ids=("X9",)
        )


def test_naive_retrieved_at_raises():
    with pytest.raises(SyntheticEvidenceError, match="timezone"):
        load_synthetic_evidence(
            EVIDENCE_FILE, retrieved_at=datetime(2026, 1, 1)
        )


def test_overlong_file_name_raises(tmp_path):
    path = write_json(
        tmp_path / f"{'a' * 120}.json", {"E1": "a"}
    )

    with pytest.raises(SyntheticEvidenceError, match="too long"):
        load_synthetic_evidence(path)


# --- Integration with the existing chunker -------------------------------


def test_chunking_preserves_evidence_identity():
    documents = load_synthetic_evidence(EVIDENCE_FILE)
    all_chunk_ids: list[str] = []

    for document in documents:
        chunks = chunk_document(document)

        assert chunks
        for chunk in chunks:
            assert chunk.document_id == document.document_id
            assert (
                document.content[chunk.start_char : chunk.end_char]
                == chunk.content
            )
            all_chunk_ids.append(chunk.chunk_id)

    assert len(all_chunk_ids) == len(set(all_chunk_ids))
def test_small_chunks_keep_evidence_identity_and_relative_offsets():
    documents = load_synthetic_evidence(EVIDENCE_FILE)
    all_chunk_ids: list[str] = []

    for document in documents:
        chunks = chunk_document(
            document, max_chars=20, overlap_chars=5
        )

        # The evidence must really be split, otherwise this test proves nothing.
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(document.content)

        for chunk in chunks:
            assert chunk.document_id == document.document_id
            # Offsets are relative to this evidence's content, not the raw JSON.
            assert (
                document.content[chunk.start_char : chunk.end_char]
                == chunk.content
            )
            all_chunk_ids.append(chunk.chunk_id)

    # No chunk_id collision between E1 and E2.
    assert len(all_chunk_ids) == len(set(all_chunk_ids))