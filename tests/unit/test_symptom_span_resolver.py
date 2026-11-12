
import pytest

from ai_healthcare.agents.symptom_span_resolver import (
    resolve_present_carrier_spans,
)
from ai_healthcare.schemas.symptom import (
    SymptomExtractionOutput,
    SymptomMention,
)


def make_extraction(
    *,
    name: str,
    reported_text: str,
    polarity: str,
) -> SymptomExtractionOutput:
    symptom = SymptomMention(
        name=name,
        reported_text=reported_text,
        polarity=polarity,
        onset_text=None,
        duration_text=None,
        severity_text=None,
        body_site_text=None,
    )

    return SymptomExtractionOutput(
        status="extracted",
        symptoms=(symptom,),
        follow_up_questions=(),
    )


@pytest.mark.parametrize(
    ("name", "reported_text", "message", "expected"),
    [
        (
            "sốt",
            "bị sốt",
            "Tôi bị sốt.",
            "sốt",
        ),
        (
            "khó thở",
            "thấy khó thở",
            "Tôi thấy khó thở.",
            "khó thở",
        ),
        (
            "đau đầu",
            "đang bị đau đầu",
            "Tôi đang bị đau đầu.",
            "đau đầu",
        ),
        (
            "buồn nôn",
            "cảm thấy buồn nôn",
            "Tôi cảm thấy buồn nôn.",
            "buồn nôn",
        ),
    ],
)
def test_resolves_present_carrier_when_candidate_matches_name(
    name,
    reported_text,
    message,
    expected,
):
    extraction = make_extraction(
        name=name,
        reported_text=reported_text,
        polarity="present",
    )

    result = resolve_present_carrier_spans(extraction, message)

    assert result.symptoms[0].reported_text == expected
    assert result.symptoms[0].reported_text in message


@pytest.mark.parametrize(
    ("reported_text", "polarity"),
    [
        ("không bị sốt", "absent"),
        ("hình như bị sốt", "uncertain"),
    ],
)
def test_does_not_modify_absent_or_uncertain_spans(
    reported_text,
    polarity,
):
    extraction = make_extraction(
        name="sốt",
        reported_text=reported_text,
        polarity=polarity,
    )

    result = resolve_present_carrier_spans(
        extraction,
        f"Tôi {reported_text}.",
    )

    assert result.symptoms[0].reported_text == reported_text


def test_preserves_symptom_defining_modifier():
    extraction = make_extraction(
        name="ho",
        reported_text="ho khan",
        polarity="present",
    )

    result = resolve_present_carrier_spans(
        extraction,
        "Tôi ho khan.",
    )

    assert result.symptoms[0].reported_text == "ho khan"


def test_does_not_remove_carrier_when_candidate_does_not_match_name():
    extraction = make_extraction(
        name="ho",
        reported_text="bị ho khan",
        polarity="present",
    )

    result = resolve_present_carrier_spans(
        extraction,
        "Tôi bị ho khan.",
    )

    assert result.symptoms[0].reported_text == "bị ho khan"


def test_does_not_mutate_original_extraction():
    extraction = make_extraction(
        name="sốt",
        reported_text="bị sốt",
        polarity="present",
    )

    resolve_present_carrier_spans(
        extraction,
        "Tôi bị sốt.",
    )

    assert extraction.symptoms[0].reported_text == "bị sốt"
