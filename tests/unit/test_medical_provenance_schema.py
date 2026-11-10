from datetime import date

import pytest
from pydantic import ValidationError

from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorSearchResult


MEDLINEPLUS_DEFAULTS = {
    "document_id": "doc_medlineplus_heart_diseases",
    "title": "Heart Diseases",
    "source_type": "health_topic_summary",
    "medical_domain": "cardiovascular",
    "source_name": "MedlinePlus, National Library of Medicine",
    "source_url": "https://medlineplus.gov/heartdiseases.html",
    "source_last_updated": "2025-03-20",
    "publication_date": None,
    "language": "en",
    "license": (
        "Public domain MedlinePlus Health Topic Summary; "
        "attribution requested: Source: MedlinePlus, National Library of Medicine."
    ),
    "content": "Heart disease is a general term that includes many types of heart problems.",
}


def test_medical_document_stores_domain_and_source_update_date_separately():
    document = MedicalDocument(**MEDLINEPLUS_DEFAULTS)

    assert document.medical_domain == "cardiovascular"
    assert document.source_last_updated == date(2025, 3, 20)
    assert document.publication_date is None


def test_health_topic_summary_requires_source_last_updated():
    data = dict(MEDLINEPLUS_DEFAULTS)
    data.pop("source_last_updated")

    with pytest.raises(ValidationError, match="source_last_updated"):
        MedicalDocument(**data)


def test_health_topic_summary_requires_explicit_domain():
    data = dict(MEDLINEPLUS_DEFAULTS)
    data.pop("medical_domain")

    with pytest.raises(ValidationError, match="medical_domain"):
        MedicalDocument(**data)


def test_health_topic_summary_requires_medlineplus_host():
    data = dict(MEDLINEPLUS_DEFAULTS)
    data["source_url"] = "https://example.org/heart-disease"

    with pytest.raises(ValidationError, match="restricted to MedlinePlus"):
        MedicalDocument(**data)


def test_health_topic_summary_requires_source_url():
    data = dict(MEDLINEPLUS_DEFAULTS)
    data["source_url"] = None

    with pytest.raises(ValidationError, match="source_url"):
        MedicalDocument(**data)


def test_existing_synthetic_document_defaults_to_other_domain():
    document = MedicalDocument(
        document_id="doc_synthetic_test",
        title="Synthetic fixture",
        source_type="synthetic_fixture",
        source_name="Unit test",
        language="vi",
        content="Synthetic content for a unit test.",
    )

    assert document.medical_domain == "other"
    assert document.source_last_updated is None


def test_medical_domain_filter_accepts_multiple_domains():
    filters = VectorSearchFilter(
        medical_domains=("cardiovascular", "renal")
    )

    assert filters.medical_domains == ("cardiovascular", "renal")


def test_medical_domain_filter_rejects_duplicate_values():
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        VectorSearchFilter(
            medical_domains=("cardiovascular", "cardiovascular")
        )


def test_medical_domain_filter_rejects_unknown_domain():
    with pytest.raises(ValidationError):
        VectorSearchFilter(medical_domains=("oncology_fake",))


def test_vector_search_result_carries_domain_and_update_date():
    result = VectorSearchResult(
        point_id="00000000-0000-0000-0000-000000000001",
        score=0.81,
        document_id="doc_medlineplus_heart_diseases",
        chunk_id="chunk_heart_0000",
        chunk_index=0,
        content="Heart disease is a general term.",
        title="Heart Diseases",
        source_type="health_topic_summary",
        medical_domain="cardiovascular",
        source_last_updated="2025-03-20",
        source_name="MedlinePlus, National Library of Medicine",
        source_url="https://medlineplus.gov/heartdiseases.html",
        retrieved_at="2026-11-08T00:00:00+00:00",
        language="en",
        start_char=0,
        end_char=33,
        raw_content_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
        normalizer_version="normalizer-v1",
        chunker_version="char-chunker-v1",
        pipeline_fingerprint="c" * 64,
        embedding_model="bge-m3",
        embedding_dimensions=1024,
    )

    assert result.medical_domain == "cardiovascular"
    assert result.source_last_updated == date(2025, 3, 20)


def test_legacy_vector_search_result_defaults_to_other_domain():
    result = VectorSearchResult(
        point_id="00000000-0000-0000-0000-000000000002",
        score=0.5,
        document_id="doc_legacy_fixture",
        chunk_id="chunk_legacy_0000",
        chunk_index=0,
        content="Legacy content.",
        title="Legacy title",
        source_type="synthetic_fixture",
        source_name="Legacy fixture",
        retrieved_at="2026-11-08T00:00:00+00:00",
        language="vi",
        start_char=0,
        end_char=15,
        raw_content_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
        normalizer_version="normalizer-v1",
        chunker_version="char-chunker-v1",
        pipeline_fingerprint="c" * 64,
        embedding_model="bge-m3",
        embedding_dimensions=1024,
    )

    assert result.medical_domain == "other"
    assert result.source_last_updated is None
