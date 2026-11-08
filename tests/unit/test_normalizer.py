import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_healthcare.ingestion.chunker import chunk_document
from ai_healthcare.ingestion.normalizer import (
    NORMALIZER_VERSION,
    NormalizationError,
    normalize_document,
    normalize_text,
)
from ai_healthcare.ingestion.synthetic_evidence_loader import (
    load_synthetic_evidence,
)
from ai_healthcare.schemas.document import MedicalDocument


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_FILE = (
    ROOT / "data" / "synthetic" / "evidence" / "latex_allergy.json"
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Generate NFC/NFD forms independently of source-file encoding.
_VIETNAMESE = "Bệnh nhân dị ứng latex"
NFC_TEXT = unicodedata.normalize("NFC", _VIETNAMESE)
NFD_TEXT = unicodedata.normalize("NFD", _VIETNAMESE)


def make_document(content: str, **overrides) -> MedicalDocument:
    fields = {
        "document_id": "doc_test-normalizer",
        "title": "Test document",
        "source_type": "other",
        "source_name": "unit-test",
        "language": "vi",
        "retrieved_at": FIXED_TIME,
        "content": content,
    }
    fields.update(overrides)
    return MedicalDocument(**fields)


# --- Unicode NFC ----------------------------------------------------------


def test_nfd_text_becomes_nfc():
    assert NFD_TEXT != NFC_TEXT
    assert len(NFD_TEXT) > len(NFC_TEXT)

    result = normalize_text(NFD_TEXT)

    assert result == NFC_TEXT
    assert unicodedata.is_normalized("NFC", result)


def test_nfc_text_is_unchanged():
    assert normalize_text(NFC_TEXT) == NFC_TEXT


def test_nfc_composition_does_not_cross_preserved_zwsp():
    text = "e\u200b\u0323\u0302"

    result = normalize_text(text)

    assert result == unicodedata.normalize("NFC", text)
    assert "\u200b" in result
    assert unicodedata.is_normalized("NFC", result)

    # ZWSP is preserved, so combining marks must not compose
    # across this character.
    assert result != "\u1ec7"


# --- Line endings ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\nb", "a\nb"),
        ("a\r\nb\rc\nd", "a\nb\nc\nd"),
        ("a\r\n\r\nb", "a\n\nb"),
        ("a\r\r\nb", "a\n\nb"),
    ],
)
def test_line_endings_are_normalized_to_lf(raw, expected):
    assert normalize_text(raw) == expected


# --- BOM ------------------------------------------------------------------


def test_leading_bom_is_removed():
    assert normalize_text("\ufeffabc") == "abc"


def test_repeated_leading_bom_is_removed():
    assert normalize_text("\ufeff\ufeffabc") == "abc"


def test_bom_in_the_middle_is_preserved():
    text = "ab\ufeffcd"

    assert normalize_text(text) == text


def test_bom_after_preserved_zwsp_is_not_removed():
    text = "\u200b\ufeffabc"

    assert normalize_text(text) == text


# --- Zero-width and invisible characters --------------------------------


@pytest.mark.parametrize(
    "char",
    [
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u2060",  # WORD JOINER
        "\u00ad",  # SOFT HYPHEN
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / internal BOM
    ],
)
def test_invisible_characters_are_preserved_inside_text(char):
    text = f"a{char}b"

    assert normalize_text(text) == text


def test_multiple_zwsp_characters_are_preserved():
    text = "a\u200b\u200bb"

    assert normalize_text(text) == text


# --- Things that must NOT change -----------------------------------------


def test_whitespace_is_not_collapsed():
    text = "a  b\t\tc\n\n\nd  \n"

    assert normalize_text(text) == text


def test_case_and_diacritics_are_not_changed():
    text = unicodedata.normalize("NFC", "ALT Bệnh NHÂN")

    assert normalize_text(text) == text


# --- Idempotency ----------------------------------------------------------


IDEMPOTENCE_INPUTS = [
    "",
    "plain",
    NFC_TEXT,
    NFD_TEXT,
    "a\r\r\nb\rc",
    "\ufeff\ufeffabc",
    "\u200b\ufeffabc",
    "\ufeff\u200b\ufeffabc",
    "e\u200b\u0323\u0302",
    "a\u200b\u200bb",
    "ab\ufeffcd",
    f"\ufeff{NFD_TEXT}\r\n\u200b",
]


@pytest.mark.parametrize("raw", IDEMPOTENCE_INPUTS)
def test_normalize_text_is_idempotent(raw):
    once = normalize_text(raw)

    assert normalize_text(once) == once


def test_normalize_text_requires_str():
    with pytest.raises(TypeError):
        normalize_text(123)  # type: ignore[arg-type]


def test_normalize_document_returns_new_document_with_normalized_content():
    document = make_document(f"\ufeff{NFD_TEXT}\r\nline two")

    result = normalize_document(document)

    assert isinstance(result, MedicalDocument)
    assert result is not document
    assert result.content == f"{NFC_TEXT}\nline two"


def test_original_document_is_not_mutated():
    raw = f"\ufeff{NFD_TEXT}\r\nline two"
    document = make_document(raw)
    snapshot = document.model_dump()

    result = normalize_document(document)

    assert document.content == raw
    assert document.model_dump() == snapshot
    assert result.content != raw


def test_all_other_fields_are_preserved():
    document = make_document(
        f"\ufeff{NFD_TEXT}",
        document_id="doc_meta-1",
        title="Guideline title",
        source_type="clinical_guideline",
        source_name="Test publisher",
        source_url="https://example.org/guideline",
        doi="10.1000/test",
        publication_date=date(2025, 1, 2),
        license="CC-BY-4.0",
        language="en",
        retrieved_at=datetime(
            2026,
            1,
            1,
            7,
            0,
            tzinfo=timezone(timedelta(hours=7)),
        ),
    )

    result = normalize_document(document)

    assert result.model_dump(exclude={"content"}) == (
        document.model_dump(exclude={"content"})
    )

    assert result.document_id == "doc_meta-1"
    assert result.doi == "10.1000/test"
    assert result.publication_date == date(2025, 1, 2)
    assert result.source_url is not None


def test_normalize_document_is_idempotent():
    document = make_document(
        f"\ufeff{NFD_TEXT}\r\n\u200bline"
    )

    once = normalize_document(document)
    twice = normalize_document(once)

    assert twice == once


def test_already_normalized_document_returns_new_equal_object():
    document = make_document(f"{NFC_TEXT}\nline two")

    result = normalize_document(document)

    assert result is not document
    assert result == document


def test_content_with_only_formatting_characters_is_rejected():
    # The characters remain preserved by normalize_text.
    # However, they do not constitute usable document text.
    document = make_document("\u200b\ufeff")

    with pytest.raises(NormalizationError, match="empty"):
        normalize_document(document)


def test_normalize_document_requires_medical_document():
    with pytest.raises(TypeError):
        normalize_document("not a document")  # type: ignore[arg-type]


# --- Real fixture ---------------------------------------------------------


def test_real_fixture_is_not_substantively_changed():
    documents = load_synthetic_evidence(
        EVIDENCE_FILE,
        required_ids=("E1", "E2"),
    )

    for document in documents:
        result = normalize_document(document)

        assert result.content == unicodedata.normalize(
            "NFC",
            document.content,
        )

        assert result.document_id == document.document_id
        assert result.source_type == "synthetic_fixture"
        assert "\r" not in result.content
        assert not result.content.startswith("\ufeff")

        # The normalizer must not remove ZWSP if it exists in the source.
        if "\u200b" in document.content:
            assert "\u200b" in result.content



def test_chunk_offsets_refer_to_normalized_content():
    raw = "\r\n".join([f"{NFD_TEXT}."] * 10)

    document = make_document(raw)
    normalized = normalize_document(document)

    # Normalization changes both character composition and line endings.
    assert normalized.content != document.content
    assert len(normalized.content) != len(document.content)

    chunks = chunk_document(
        normalized,
        max_chars=30,
        overlap_chars=5,
    )

    assert len(chunks) > 1
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(normalized.content)

    for chunk in chunks:
        assert chunk.document_id == normalized.document_id

        assert (
            normalized.content[chunk.start_char:chunk.end_char]
            == chunk.content
        )

        assert "\r" not in chunk.content
        assert unicodedata.is_normalized("NFC", chunk.content)

def test_normalizer_version_is_a_non_empty_string():
    assert isinstance(NORMALIZER_VERSION, str)
    assert NORMALIZER_VERSION.strip()
