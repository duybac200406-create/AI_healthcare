import json
import string
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from ai_healthcare.ingestion.normalizer import normalize_document
from ai_healthcare.ingestion.structured_chunker import (
    HEADING_MAX_CHARS,
    STRUCTURED_CHUNKER_VERSION,
    chunk_document_structured,
)
from ai_healthcare.ingestion.structured_indexer import StructuredDocumentIndexer
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.ingestion import IngestionRecord
from tests.unit.test_document_indexer import (
    FakeEmbeddingProvider,
    FakeVectorStore,
    make_document,
)

CORPUS = (
    Path(__file__).resolve().parents[2]
    / "data" / "samples" / "medlineplus_symptom_4b_lite_v1"
)

_word = st.text(alphabet=string.ascii_letters, min_size=1, max_size=10)


@st.composite
def _paragraph(draw, max_sentences=10):
    sentences = draw(
        st.lists(
            st.lists(_word, min_size=1, max_size=12).map(" ".join),
            min_size=1,
            max_size=max_sentences,
        )
    )
    return ". ".join(sentences) + "."


@st.composite
def _section(draw, max_sentences=10):
    lines = []
    if draw(st.booleans()):
        words = draw(st.lists(_word, min_size=1, max_size=8))
        lines.append(" ".join(words) + "?")
    body = draw(
        st.lists(
            st.one_of(
                _paragraph(max_sentences),
                st.lists(_word, min_size=1, max_size=12).map(
                    lambda w: "- " + " ".join(w)
                ),
            ),
            min_size=1,
            max_size=6,
        )
    )
    return lines + body


@st.composite
def contents(draw, max_sentences=10):
    sections = draw(st.lists(_section(max_sentences), min_size=1, max_size=8))
    lines = [line for section in sections for line in section]
    separator = draw(st.sampled_from(["\n", "\n\n"]))
    return separator.join(lines)


def _is_heading(line):
    s = line.strip()
    return s.endswith("?") and len(s) <= HEADING_MAX_CHARS and not s.startswith("- ")


def _check_invariants(source, chunks, max_chars, *, check_clean):
    assert chunks, "a non-blank document must produce chunks"

    covered = [0] * len(source)
    previous_end = 0
    for index, c in enumerate(chunks):
        assert c.chunk_index == index
        assert c.content == source[c.start_char : c.end_char]
        assert len(c.content) <= max_chars
        assert c.start_char >= previous_end, "chunks must not overlap"
        previous_end = c.end_char
        for i in range(c.start_char, c.end_char):
            covered[i] += 1

        last_line = c.content.rstrip().split("\n")[-1]
        assert not _is_heading(last_line), "a heading must not end a chunk"

        if check_clean:
            assert c.start_char == 0 or source[c.start_char - 1] == "\n"
            assert c.end_char == len(source) or source[c.end_char] == "\n"

    for i, ch in enumerate(source):
        if not ch.isspace():
            assert covered[i] == 1, f"character {i} is not covered exactly once"


@settings(max_examples=150, deadline=None)
@given(contents(), st.sampled_from([300, 500, 1200]))
def test_chunks_are_exact_ordered_complete_and_bounded(content, max_chars):
    doc = make_document(content)
    chunks = chunk_document_structured(doc, max_chars=max_chars)
    unit_limit = max_chars - HEADING_MAX_CHARS - 1
    clean = all(len(line) <= unit_limit for line in content.split("\n"))
    _check_invariants(doc.content, chunks, max_chars, check_clean=clean)
    # Deterministic.
    assert chunks == chunk_document_structured(doc, max_chars=max_chars)


@settings(max_examples=60, deadline=None)
@given(contents(max_sentences=60), st.sampled_from([300, 500, 1200]))
def test_very_long_lines_are_split_without_losing_text(content, max_chars):
    doc = make_document(content)
    chunks = chunk_document_structured(doc, max_chars=max_chars)
    _check_invariants(doc.content, chunks, max_chars, check_clean=False)


def test_a_section_that_fits_is_never_split():
    body = "What is it?\n" + "Sentence one is here. " * 5 + "\n- item a\n- item b"
    doc = make_document(body)
    chunks = chunk_document_structured(doc, max_chars=1200)
    assert len(chunks) == 1
    assert chunks[0].section_title == "What is it?"


def test_section_title_follows_the_section_into_continuation_chunks():
    long_line = "Long sentence number one is here. " * 30
    doc = make_document(f"Why does it happen?\n{long_line}")
    chunks = chunk_document_structured(doc, max_chars=300)
    assert len(chunks) > 1
    assert all(c.section_title == "Why does it happen?" for c in chunks)


def test_rejects_unusably_small_limit():
    with pytest.raises(ValueError):
        chunk_document_structured(make_document("Some text here."), max_chars=100)


@pytest.mark.parametrize("path", sorted(CORPUS.glob("*.json")), ids=lambda p: p.stem)
def test_real_corpus_documents_chunk_cleanly_and_fit_the_ingestion_record(path):
    raw = MedicalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
    doc = normalize_document(raw)
    chunks = chunk_document_structured(doc, max_chars=1200)
    _check_invariants(doc.content, chunks, 1200, check_clean=True)

    record = IngestionRecord.from_documents(
        raw_document=raw,
        normalized_document=doc,
        chunks=chunks,
        normalizer_version="normalizer-v1",
        chunker_version=STRUCTURED_CHUNKER_VERSION,
        max_chars=1200,
        overlap_chars=0,
    )
    assert record.chunk_count == len(chunks)


def test_structured_indexer_records_its_own_chunker_version():
    store = FakeVectorStore()
    indexer = StructuredDocumentIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        max_chars=300,
        batch_size=2,
    )
    content = "\n".join(
        f"Question number {i}?\n" + f"Answer sentence {i}. " * 12 for i in range(4)
    )
    ingestion = indexer.index_document(make_document(content))

    assert ingestion.chunker_version == STRUCTURED_CHUNKER_VERSION
    assert ingestion.overlap_chars == 0
    assert len(store.records) == ingestion.chunk_count > 1
    assert all(r.chunk.section_title for r in store.records)
