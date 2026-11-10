import json
from datetime import datetime, timezone
from pathlib import Path

from ai_healthcare.schemas.document import MedicalDocument


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data" / "samples" / "medlineplus_latex_allergy.json"


def test_medlineplus_fixture_has_traceable_public_domain_source():
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    metadata = payload["source_metadata"]
    document_payload = dict(payload["document"])
    document_payload.setdefault("retrieved_at", datetime.now(timezone.utc))
    document = MedicalDocument.model_validate(document_payload)

    assert metadata["page_last_updated"] == "2025-11-30"
    assert metadata["copyright_status"] == "public_domain"
    assert metadata["attribution"] == "Source: MedlinePlus, National Library of Medicine."
    assert document.title == "Latex Allergy"
    assert document.source_name == "MedlinePlus, National Library of Medicine"
    assert str(document.source_url) == "https://medlineplus.gov/latexallergy.html"
    assert document.source_type == "other"
    assert document.publication_date is None
    assert document.language == "en"
    assert "physical exam and skin and blood tests" in document.content


def test_medlineplus_fixture_is_not_marked_as_synthetic_or_guideline():
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    document_payload = dict(payload["document"])
    document_payload.setdefault("retrieved_at", datetime.now(timezone.utc))
    document = MedicalDocument.model_validate(document_payload)

    assert document.source_type not in {"synthetic_fixture", "clinical_guideline"}
