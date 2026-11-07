import pytest
from pydantic import ValidationError

from ai_healthcare.ingestion.chunker import chunk_document
from ai_healthcare.schemas.document import MedicalDocument


def make_document(content: str) -> MedicalDocument:
    return MedicalDocument(
        document_id="doc_synthetic_latex_001",
        title="Synthetic Latex Allergy Evidence",
        source_type="synthetic_fixture",
        source_name="AI Healthcare synthetic test data",
        language="vi",
        content=content,
    )


def test_short_document_produces_one_chunk() -> None:
    document = make_document("Đây là tài liệu giả lập.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_long_document_is_split_within_max_chars() -> None:
    content = " ".join(
        f"Medical sentence number {i}."
        for i in range(30)
    )

    document = make_document(content)

    chunks = chunk_document(
        document,
        max_chars=100,
        overlap_chars=20,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk.content) <= 100
        for chunk in chunks
    )


def test_chunk_offsets_match_original_document() -> None:
    content = "First medical paragraph. Second medical paragraph."

    document = make_document(content)

    chunks = chunk_document(
        document,
        max_chars=30,
        overlap_chars=5,
    )

    for chunk in chunks:
        assert document.content[
            chunk.start_char:chunk.end_char
        ] == chunk.content


def test_chunk_metadata_is_preserved() -> None:
    document = make_document("Medical evidence fixture.")

    chunks = chunk_document(document)

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.language == document.language
    assert chunk.chunk_index == 0
    assert chunk.chunk_id.startswith("chunk_")


def test_chunk_ids_are_unique() -> None:
    document = make_document(
        " ".join(f"word{i}" for i in range(100))
    )

    chunks = chunk_document(
        document,
        max_chars=50,
        overlap_chars=5,
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.parametrize("max_chars", [0, -1])
def test_invalid_max_chars_is_rejected(max_chars: int) -> None:
    document = make_document("Medical evidence.")

    with pytest.raises(ValueError, match="greater than zero"):
        chunk_document(document, max_chars=max_chars)


@pytest.mark.parametrize("overlap_chars", [-1, 100])
def test_invalid_overlap_is_rejected(overlap_chars: int) -> None:
    document = make_document("Medical evidence.")

    with pytest.raises(ValueError):
        chunk_document(
            document,
            max_chars=100,
            overlap_chars=overlap_chars,
        )


def test_non_medical_document_is_rejected() -> None:
    with pytest.raises(TypeError, match="MedicalDocument"):
        chunk_document("Not a medical document")  # type: ignore[arg-type]


def test_blank_document_is_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        make_document("   ")
