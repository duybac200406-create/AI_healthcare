import pytest
from pydantic import ValidationError

from ai_healthcare.schemas.document import (
    DocumentChunk,
    MedicalDocument,
)


def make_document(**overrides):
    data = {
        "document_id": "doc_latex_001",
        "title": "Latex Allergy Clinical Reference",
        "source_type": "clinical_guideline",
        "source_name": "Example Medical Source",
        "source_url": "https://example.org/latex",
        "language": "en",
        "content": "Latex allergy is an immune response.",
    }

    data.update(overrides)

    return data


def make_chunk(**overrides):
    data = {
        "chunk_id": "chunk_latex_001",
        "document_id": "doc_latex_001",
        "chunk_index": 0,
        "content": "Latex allergy is an immune response.",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "start_char": 0,
        "end_char": 38,
    }

    data.update(overrides)

    return data


def test_valid_medical_document() -> None:
    document = MedicalDocument(**make_document())

    assert document.document_id == "doc_latex_001"
    assert document.source_type == "clinical_guideline"
    assert document.retrieved_at.utcoffset() is not None


def test_invalid_document_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MedicalDocument(
            **make_document(source_type="unknown_source")
        )


def test_empty_document_content_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MedicalDocument(**make_document(content="   "))


def test_valid_document_chunk() -> None:
    chunk = DocumentChunk(**make_chunk())

    assert chunk.chunk_index == 0
    assert chunk.document_id == "doc_latex_001"
    assert chunk.page_start == 1
    assert chunk.end_char > chunk.start_char


def test_page_range_requires_both_boundaries() -> None:
    with pytest.raises(
        ValidationError,
        match="provided together",
    ):
        DocumentChunk(
            **make_chunk(page_end=None)
        )


def test_character_range_must_be_valid() -> None:
    with pytest.raises(
        ValidationError,
        match="greater than start_char",
    ):
        DocumentChunk(
            **make_chunk(start_char=20, end_char=10)
        )


def test_embedding_field_is_not_allowed_in_document_chunk() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            **make_chunk(embedding=[0.1, 0.2, 0.3])
        )
