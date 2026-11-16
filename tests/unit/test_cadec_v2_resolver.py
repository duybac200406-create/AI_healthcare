
from scripts.evaluate_cadec_v2 import (
    PredictedMention,
    find_all_occurrences,
    resolve_mention_offset,
)


def test_resolves_unique_exact_occurrence():
    source = "Patient reports nausea after medication."

    mention = PredictedMention(
        text="nausea",
        left_context="incorrect context",
        right_context="incorrect context",
    )

    result = resolve_mention_offset(mention, source)

    assert result["resolved"] is True
    assert result["resolution_status"] == "unique_exact_text"
    assert source[result["start_char"]:result["end_char"]] == "nausea"
    assert result["occurrence_count"] == 1


def test_repeated_surface_resolved_by_exact_context():
    source = "Patient had pain yesterday. Today she reports pain again."

    start = source.rfind("pain")
    end = start + len("pain")

    mention = PredictedMention(
        text="pain",
        left_context=source[max(0, start - 32):start],
        right_context=source[end:end + 32],
    )

    result = resolve_mention_offset(mention, source)

    assert result["resolved"] is True
    assert result["resolution_status"] == (
        "repeated_text_disambiguated_by_context"
    )
    assert result["start_char"] == start
    assert result["end_char"] == end
    assert result["occurrence_count"] == 2


def test_repeated_surface_with_mismatched_context_is_unresolved():
    source = "pain in left arm; pain in right arm"

    mention = PredictedMention(
        text="pain",
        left_context="not present in source",
        right_context="not present in source",
    )

    result = resolve_mention_offset(mention, source)

    assert result["resolved"] is False
    assert result["resolution_status"] == "context_mismatch"
    assert result["start_char"] is None
    assert result["end_char"] is None


def test_repeated_surface_with_multiple_context_matches_is_ambiguous():
    prefix = "L" * 32
    suffix = "R" * 32

    source = (
        prefix + "pain" + suffix
        + prefix + "pain" + suffix
    )

    mention = PredictedMention(
        text="pain",
        left_context=prefix,
        right_context=suffix,
    )

    result = resolve_mention_offset(mention, source)

    assert result["resolved"] is False
    assert result["resolution_status"] == "ambiguous_context"
    assert result["occurrence_count"] == 2
    assert result["context_candidate_count"] == 2


def test_surface_not_found_is_unresolved():
    source = "Patient reports nausea."

    mention = PredictedMention(
        text="headache",
        left_context="Patient reports ",
        right_context=".",
    )

    result = resolve_mention_offset(mention, source)

    assert result["resolved"] is False
    assert result["resolution_status"] == "text_not_found_exactly"
    assert result["occurrence_count"] == 0


def test_finds_overlapping_occurrences():
    assert find_all_occurrences("ababa", "aba") == [0, 2]
