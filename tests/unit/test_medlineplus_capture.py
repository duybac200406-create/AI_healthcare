from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from ai_healthcare.ingestion.medlineplus_capture import (
    MEDLINEPLUS_ORGANIZATION,
    MedlinePlusCaptureError,
    _parse_service_response,
    capture_health_topic,
)


SPEC = {
    "document_id": "doc_medlineplus_heart_diseases",
    "title": "Heart Diseases",
    "url": "https://medlineplus.gov/heartdiseases.html",
    "medical_domain": "cardiovascular",
    "source_type": "health_topic_summary",
    "language": "en",
    "source_last_updated": "2025-03-20",
}


def xml_response(*, url=None, title="Heart Diseases", organization=MEDLINEPLUS_ORGANIZATION, summary=None):
    url = url or SPEC["url"]
    summary = summary or (
        "<p>Heart disease is a group of conditions affecting the heart.</p>"
        "<p>Risk factors can include high blood pressure.</p>"
        "<ul><li>Example factor one</li><li>Example factor two</li></ul>"
    )
    return f"""<?xml version='1.0' encoding='UTF-8'?>
    <nlmSearchResult><list><document url='{url}' rank='0'>
      <content name='title'>{title}</content>
      <content name='organizationName'>{organization}</content>
      <content name='FullSummary'>{summary}</content>
    </document></list></nlmSearchResult>"""


def make_client(body: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("db") == "healthTopics"
        assert request.url.params.get("rettype") == "brief"
        return httpx.Response(200, text=body, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_capture_returns_document_with_provenance():
    with make_client(xml_response()) as client:
        document = capture_health_topic(
            SPEC,
            client=client,
            retrieved_at=datetime(2026, 11, 8, tzinfo=timezone.utc),
        )

    assert document.document_id == SPEC["document_id"]
    assert document.title == "Heart Diseases"
    assert str(document.source_url) == SPEC["url"]
    assert document.source_type == "health_topic_summary"
    assert document.medical_domain == "cardiovascular"
    assert document.source_last_updated.isoformat() == "2025-03-20"
    assert document.publication_date is None
    assert "Heart disease is a group" in document.content
    assert "- Example factor one" in document.content
    assert document.retrieved_at == datetime(2026, 11, 8, tzinfo=timezone.utc)


def test_capture_rejects_similar_but_wrong_topic_url():
    body = xml_response(url="https://medlineplus.gov/diabetes.html")
    with make_client(body) as client:
        with pytest.raises(MedlinePlusCaptureError, match="exactly one matching"):
            capture_health_topic(SPEC, client=client)


def test_capture_rejects_wrong_title():
    with make_client(xml_response(title="Different Topic")) as client:
        with pytest.raises(MedlinePlusCaptureError, match="title"):
            capture_health_topic(SPEC, client=client)


def test_capture_rejects_wrong_organization():
    with make_client(xml_response(organization="Third Party Organization")) as client:
        with pytest.raises(MedlinePlusCaptureError, match="organization"):
            capture_health_topic(SPEC, client=client)


def test_capture_rejects_missing_summary():
    body = """<nlmSearchResult><list><document url='https://medlineplus.gov/heartdiseases.html'>
    <content name='title'>Heart Diseases</content>
    <content name='organizationName'>National Library of Medicine</content>
    </document></list></nlmSearchResult>"""
    with make_client(body) as client:
        with pytest.raises(MedlinePlusCaptureError, match="no FullSummary"):
            capture_health_topic(SPEC, client=client)


def test_capture_rejects_unexpected_or_invalid_url():
    bad = dict(SPEC, url="https://example.com/heartdiseases.html")
    with make_client(xml_response()) as client:
        with pytest.raises(MedlinePlusCaptureError, match="HTTPS MedlinePlus"):
            capture_health_topic(bad, client=client)


def test_summary_parser_preserves_paragraph_and_list_boundaries():
    title, text = _parse_service_response(
        xml_response(),
        expected_url=SPEC["url"],
        expected_title=SPEC["title"],
    )
    assert title == "Heart Diseases"
    assert "\n" in text
    assert "- Example factor one" in text
    assert "- Example factor two" in text


def test_manifest_source_last_updated_is_not_retrieved_at():
    with make_client(xml_response()) as client:
        document = capture_health_topic(
            SPEC,
            client=client,
            retrieved_at=datetime(2026, 11, 8, tzinfo=timezone.utc),
        )
    assert document.source_last_updated.isoformat() == "2025-03-20"
    assert document.retrieved_at.year == 2026


def test_capture_rejects_naive_retrieved_at():
    with make_client(xml_response()) as client:
        with pytest.raises(ValueError, match="timezone-aware"):
            capture_health_topic(SPEC, client=client, retrieved_at=datetime(2026, 11, 8))


def test_capture_rejects_invalid_xml():
    with make_client("<not-xml") as client:
        with pytest.raises(MedlinePlusCaptureError, match="invalid XML"):
            capture_health_topic(SPEC, client=client)


def test_multidomain_manifest_has_seven_distinct_topics_and_domains():
    import json
    from pathlib import Path

    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "manifests"
        / "medlineplus_multidomain_pilot_v0.2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["sources"]

    assert len(sources) == 7
    assert len({item["document_id"] for item in sources}) == 7
    assert len({item["medical_domain"] for item in sources}) == 7
    assert all(item["source_type"] == "health_topic_summary" for item in sources)
    assert all(item["url"].startswith("https://medlineplus.gov/") for item in sources)


def test_manifest_uses_specific_type2_diabetes_topic_not_umbrella_diabetes():
    import json
    from pathlib import Path

    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "manifests"
        / "medlineplus_multidomain_pilot_v0.2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diabetes = next(item for item in manifest["sources"] if item["medical_domain"] == "endocrine_metabolic")

    assert diabetes["title"] == "Diabetes Type 2"
    assert diabetes["url"] == "https://medlineplus.gov/diabetestype2.html"
    assert diabetes["source_last_updated"] == "2026-01-21"

@pytest.mark.parametrize(
    ("highlighted_title", "visible_title"),
    [
        ('&lt;span class="qt0"&gt;Heart Diseases&lt;/span&gt;', "Heart Diseases"),
        (
            '&lt;span class="qt0"&gt;Heart &amp;amp; Vascular Diseases&lt;/span&gt;',
            "Heart & Vascular Diseases",
        ),
        (
            '&lt;span class="qt0"&gt;Heart &lt;em&gt;Diseases&lt;/em&gt;&lt;/span&gt;',
            "Heart Diseases",
        ),
    ],
)
def test_capture_strips_web_service_title_highlighting_before_exact_match(
    highlighted_title,
    visible_title,
):
    """Query-hit highlighting is presentation markup, not part of the title."""
    body = xml_response(title=highlighted_title)
    expected_spec = dict(SPEC, title=visible_title)
    with make_client(body) as client:
        document = capture_health_topic(expected_spec, client=client)

    assert document.title == visible_title


def test_capture_still_rejects_wrong_visible_title_with_highlighting():
    body = xml_response(
        title='&lt;span class="qt0"&gt;Heart Failure&lt;/span&gt;'
    )
    with make_client(body) as client:
        with pytest.raises(MedlinePlusCaptureError, match="title"):
            capture_health_topic(SPEC, client=client)

def test_extract_summary_markup_removes_html_and_preserves_structure():
    import re
    import xml.etree.ElementTree as ET
    from xml.sax.saxutils import escape

    from ai_healthcare.ingestion.medlineplus_capture import (
        _extract_summary_markup,
    )

    html = (
        '<p>Kidney <span class="qt0">disease</span> develops slowly.</p>'
        '<ul>'
        '<li>Blood &amp; urine tests</li>'
        '<li>Other item</li>'
        '</ul>'
        '<p>End.</p>'
    )

    xml = (
        '<content name="FullSummary">'
        + escape(html)
        + '</content>'
    )

    element = ET.fromstring(xml)
    result = _extract_summary_markup(element)

    assert result == (
        "Kidney disease develops slowly.\n"
        "- Blood & urine tests\n"
        "- Other item\n"
        "End."
    )

    assert not re.search(r"</?[A-Za-z][^>]*>", result)
    assert "&amp;" not in result