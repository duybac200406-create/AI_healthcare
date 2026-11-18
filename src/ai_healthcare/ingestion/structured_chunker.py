"""Structure-aware chunker for heading-based plain-text documents.

Deterministic and model-free. MedlinePlus health-topic text is a sequence of
lines: short question headings ("What causes X?"), paragraphs and "- " list
items. This chunker:

* treats every line as an atomic unit and every heading plus the lines that
  follow it as one section;
* packs WHOLE sections into a chunk up to ``max_chars`` and never splits a
  section that fits;
* splits an oversize section at line boundaries, an oversize line at sentence
  boundaries, and only as a last resort at a word boundary;
* never leaves a heading as the last line of a chunk;
* keeps ``content`` an exact slice of the source (``start_char``/``end_char``
  stay faithful) and records the section heading in ``section_title``;
* uses no overlap, because sections are self-contained.

The original character chunker (``chunker.py``) is unchanged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ai_healthcare.schemas.document import DocumentChunk, MedicalDocument

STRUCTURED_CHUNKER_VERSION = "structured-chunker-v1"

HEADING_MAX_CHARS = 120
MIN_MAX_CHARS = 300

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    is_heading: bool


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.endswith("?")
        and len(stripped) <= HEADING_MAX_CHARS
        and not stripped.startswith("- ")
    )


def _split_long_line(
    source: str, start: int, end: int, limit: int
) -> list[tuple[int, int]]:
    """Split one long line into spans of at most ``limit`` characters."""

    spans: list[tuple[int, int]] = []

    # Sentence-level pieces first.
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_BREAK.finditer(source, start, end):
        pieces.append((cursor, match.start()))
        cursor = match.end()
    pieces.append((cursor, end))

    current_start: int | None = None
    current_end = start
    for p_start, p_end in pieces:
        while p_end - p_start > limit:  # sentence longer than the limit
            cut = p_start + limit
            boundary = cut
            while boundary > p_start and not source[boundary - 1].isspace():
                boundary -= 1
            if boundary <= p_start:
                boundary = cut
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
            spans.append((p_start, boundary))
            p_start = boundary
            while p_start < p_end and source[p_start].isspace():
                p_start += 1
        if p_start >= p_end:
            continue
        if current_start is None:
            current_start, current_end = p_start, p_end
        elif p_end - current_start <= limit:
            current_end = p_end
        else:
            spans.append((current_start, current_end))
            current_start, current_end = p_start, p_end
    if current_start is not None:
        spans.append((current_start, current_end))

    # Trailing whitespace never belongs to a span.
    cleaned: list[tuple[int, int]] = []
    for s_start, s_end in spans:
        while s_end > s_start and source[s_end - 1].isspace():
            s_end -= 1
        if s_end > s_start:
            cleaned.append((s_start, s_end))
    return cleaned


def _build_units(source: str, unit_limit: int) -> list[_Unit]:
    units: list[_Unit] = []
    offset = 0
    for line in source.split("\n"):
        line_start = offset
        offset += len(line) + 1
        stripped_left = len(line) - len(line.lstrip())
        stripped = line.strip()
        if not stripped:
            continue
        start = line_start + stripped_left
        end = start + len(stripped)
        if end - start <= unit_limit:
            units.append(_Unit(start, end, _is_heading(stripped)))
        else:
            for s_start, s_end in _split_long_line(source, start, end, unit_limit):
                units.append(_Unit(s_start, s_end, False))
    return units


def chunk_document_structured(
    document: MedicalDocument,
    *,
    max_chars: int = 1200,
) -> tuple[DocumentChunk, ...]:
    if not isinstance(document, MedicalDocument):
        raise TypeError("document must be a MedicalDocument instance.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer.")
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}.")

    source = document.content
    # Heading plus any following unit must always fit in one chunk.
    unit_limit = max_chars - HEADING_MAX_CHARS - 1
    units = _build_units(source, unit_limit)

    # Group into sections: a heading starts a new section.
    sections: list[list[_Unit]] = []
    for unit in units:
        if unit.is_heading or not sections:
            sections.append([unit])
        else:
            sections[-1].append(unit)

    def heading_of(section: list[_Unit]) -> str | None:
        first = section[0]
        return source[first.start : first.end] if first.is_heading else None

    # Section heading of every unit (a continuation chunk keeps its section's
    # heading as ``section_title`` even though the heading text is elsewhere).
    titles: dict[int, str | None] = {}
    for section in sections:
        title = heading_of(section)
        for unit in section:
            titles[unit.start] = title

    spans: list[tuple[int, int, str | None]] = []
    current: list[_Unit] = []

    def flush() -> list[_Unit]:
        """Close the current chunk; return trailing headings to carry over."""

        nonlocal current
        carry: list[_Unit] = []
        while len(current) > 1 and current[-1].is_heading:
            carry.insert(0, current.pop())
        if current:
            spans.append(
                (current[0].start, current[-1].end, titles[current[0].start])
            )
        current = []
        return carry

    for section in sections:
        section_len = section[-1].end - section[0].start

        if section_len <= max_chars:
            if current and section[-1].end - current[0].start > max_chars:
                current = flush()
            current.extend(section)
            continue

        # Oversize section: pack line by line, continuing from whatever the
        # previous chunk left open so no tiny tail chunk is stranded.
        for unit in section:
            if current and unit.end - current[0].start > max_chars:
                current = flush()
            current.append(unit)
    flush()

    document_hash = hashlib.sha256(
        document.document_id.encode("utf-8")
    ).hexdigest()[:12]

    return tuple(
        DocumentChunk(
            chunk_id=f"chunk_{document_hash}_{index:04d}",
            document_id=document.document_id,
            chunk_index=index,
            content=source[start:end],
            section_title=title[:500] if title else None,
            language=document.language,
            start_char=start,
            end_char=end,
        )
        for index, (start, end, title) in enumerate(spans)
    )
