"""Capture only MedlinePlus Health Topic Summary content via the official Web Service.

This module deliberately does not scrape the full HTML page. It asks the official
MedlinePlus Web Service for a topic record and extracts only its FullSummary node.
The caller-supplied source_last_updated value is copied from a separately verified
Health Topic page; the Web Service record itself does not provide that page footer date.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any, Mapping
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx

from ai_healthcare.schemas.document import MedicalDocument

MEDLINEPLUS_WEB_SERVICE_URL = "https://wsearch.nlm.nih.gov/ws/query"
MEDLINEPLUS_ORGANIZATION = "National Library of Medicine"
MEDLINEPLUS_SOURCE_NAME = "MedlinePlus, National Library of Medicine"
MEDLINEPLUS_ATTRIBUTION = "Source: MedlinePlus, National Library of Medicine."
MEDLINEPLUS_PUBLIC_DOMAIN_LICENSE = "Public domain – MedlinePlus Health Topic Summary; attribution required."


class MedlinePlusCaptureError(RuntimeError):
    """Raised when an expected MedlinePlus Health Topic Summary cannot be captured."""


class _SummaryHTMLText(HTMLParser):
    """Convert summary HTML into readable text while preserving block boundaries."""

    _BLOCK_TAGS = {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "pre",
        "section", "table", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._newline()
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        normalized = re.sub(r"\s+", " ", data)
        if normalized:
            self.parts.append(normalized)

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def get_text(self) -> str:
        text = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        compact: list[str] = []
        for line in lines:
            if line:
                compact.append(line)
        return "\n".join(compact).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _html_fragment_to_plain_text(value: str) -> str:
    """Remove HTML markup from a Web Service text field without losing its text.

    MedlinePlus search responses can wrap matched title terms in highlighting
    markup, for example ``<span class="qt0">Heart Diseases</span>``. XML
    parsing decodes the escaped markup into a string, so Element.itertext()
    alone returns the literal tags. Parse that fragment as HTML and retain only
    its visible text before validating the manifest title.
    """
    parser = _SummaryHTMLText()
    parser.feed(value)
    parser.close()
    return re.sub(r"\s+", " ", parser.get_text()).strip()


def _named_content_text(parent: ET.Element, content_name: str) -> str | None:
    """Read a MedlinePlus <content name=...> value as visible text.

    The Web Service may return query-highlight spans inside title values. The
    markup is presentation-only and must not cause a valid exact-title match to
    be rejected. The visible text is still compared strictly to the manifest.
    """
    for child in parent.iter():
        if (
            _local_name(child.tag).casefold() == "content"
            and child.attrib.get("name", "").casefold() == content_name.casefold()
        ):
            raw_text = "".join(child.itertext()).strip()
            if raw_text:
                text = _html_fragment_to_plain_text(raw_text)
                if text:
                    return text
    return None


def _normalize_medlineplus_url(value: str) -> str:
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if parsed.scheme.lower() != "https" or host not in {"medlineplus.gov", "www.medlineplus.gov"}:
        raise MedlinePlusCaptureError("Expected an HTTPS MedlinePlus topic URL.")
    if parsed.query or parsed.fragment:
        raise MedlinePlusCaptureError("MedlinePlus topic URL must not contain query or fragment components.")
    return f"https://medlineplus.gov{path}"


def _extract_summary_markup(summary_element: ET.Element) -> str:

    if len(summary_element) == 0:
        # Case 1: HTML is stored as escaped XML text.
        markup = summary_element.text or ""
    else:
        # Case 2: HTML tags are actual child elements.
        # Serialize the inner content, not the FullSummary wrapper.
        parts = [summary_element.text or ""]

        for child in summary_element:
            parts.append(
                ET.tostring(
                    child,
                    encoding="unicode",
                    method="xml",
                )
            )

        markup = "".join(parts)

    parser = _SummaryHTMLText()
    parser.feed(markup)
    parser.close()

    return parser.get_text()

def _parse_service_response(
    xml_text: str,
    *,
    expected_url: str,
    expected_title: str,
) -> tuple[str, str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise MedlinePlusCaptureError("MedlinePlus Web Service returned invalid XML.") from exc

    normalized_expected_url = _normalize_medlineplus_url(expected_url)
    documents = [element for element in root.iter() if _local_name(element.tag) == "document"]
    exact_matches: list[ET.Element] = []
    for document in documents:
        candidate_url = document.attrib.get("url", "")
        try:
            normalized_candidate = _normalize_medlineplus_url(candidate_url)
        except MedlinePlusCaptureError:
            continue
        if normalized_candidate == normalized_expected_url:
            exact_matches.append(document)

    if len(exact_matches) != 1:
        raise MedlinePlusCaptureError(
            "Expected exactly one matching MedlinePlus topic URL in the Web Service response."
        )

    document = exact_matches[0]
    title = _named_content_text(document, "title")
    organization = _named_content_text(document, "organizationName")
    if not title or title.casefold() != expected_title.casefold():
        raise MedlinePlusCaptureError("MedlinePlus topic title does not match the manifest.")
    if organization != MEDLINEPLUS_ORGANIZATION:
        raise MedlinePlusCaptureError("MedlinePlus organization metadata is missing or unexpected.")

    summary_element = next(
        (
            node for node in document.iter()
            if _local_name(node.tag).casefold() == "content"
            and node.attrib.get("name", "").casefold() == "fullsummary"
        ),
        None,
    )
    if summary_element is None:
        raise MedlinePlusCaptureError("MedlinePlus response has no FullSummary content.")

    summary = _extract_summary_markup(summary_element)
    if len(summary) < 80:
        raise MedlinePlusCaptureError("MedlinePlus FullSummary is unexpectedly short.")
    return title, summary


def capture_health_topic(
    spec: Mapping[str, Any],
    *,
    client: httpx.Client,
    retrieved_at: datetime | None = None,
) -> MedicalDocument:
    """Fetch one exact MedlinePlus topic and return a validated MedicalDocument.

    source_last_updated is taken from the reviewed manifest, not inferred from the
    Web Service or current retrieval time.
    """
    required = {
        "document_id", "title", "url", "medical_domain", "source_last_updated",
        "source_type", "language",
    }
    missing = required - set(spec)
    if missing:
        raise MedlinePlusCaptureError(f"Manifest entry is missing fields: {sorted(missing)}")

    expected_url = str(spec["url"])
    _normalize_medlineplus_url(expected_url)
    expected_title = str(spec["title"]).strip()
    if not expected_title:
        raise MedlinePlusCaptureError("Manifest title must not be blank.")

    updated_value = spec["source_last_updated"]
    source_last_updated = date.fromisoformat(updated_value) if isinstance(updated_value, str) else updated_value
    if not isinstance(source_last_updated, date):
        raise MedlinePlusCaptureError("source_last_updated must be an ISO date.")

    # Search exact title and then verify the exact canonical page URL. Never
    # accept a merely similar search result or substitute another topic.
    query_term = f'title:"{expected_title}"'
    params = {
        "db": "healthTopics",
        "term": query_term,
        "rettype": "brief",
        "retmax": "20",
        "tool": "aihealthcaremedicalrag",
    }
    try:
        response = client.get(MEDLINEPLUS_WEB_SERVICE_URL, params=params)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise MedlinePlusCaptureError("MedlinePlus Web Service request timed out.") from exc
    except httpx.HTTPError as exc:
        raise MedlinePlusCaptureError("MedlinePlus Web Service request failed.") from exc

    title, summary = _parse_service_response(
        response.text,
        expected_url=expected_url,
        expected_title=expected_title,
    )

    timestamp = retrieved_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    try:
        return MedicalDocument(
            document_id=str(spec["document_id"]),
            title=title,
            source_type=str(spec["source_type"]),
            source_name=MEDLINEPLUS_SOURCE_NAME,
            source_url=expected_url,
            publication_date=None,
            retrieved_at=timestamp,
            language=str(spec["language"]),
            license=MEDLINEPLUS_PUBLIC_DOMAIN_LICENSE,
            content=summary,
            medical_domain=str(spec["medical_domain"]),
            source_last_updated=source_last_updated,
        )
    except Exception as exc:
        raise MedlinePlusCaptureError("Captured MedlinePlus source failed MedicalDocument validation.") from exc
