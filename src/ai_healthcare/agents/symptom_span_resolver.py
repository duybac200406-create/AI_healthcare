from __future__ import annotations

import unicodedata

from ai_healthcare.schemas.symptom import SymptomExtractionOutput


# Only remove a carrier when the remaining text exactly matches
# the normalized symptom name.
_PRESENT_CARRIER_PREFIXES = frozenset(
    {
        "bị",
        "đang bị",
        "thấy",
        "cảm thấy",
    }
)


def _comparison_key(value: str) -> str:
    """Normalize text for comparison without changing source spans."""
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.casefold().split())


def resolve_present_carrier_spans(
    extraction: SymptomExtractionOutput,
    user_message: str,
) -> SymptomExtractionOutput:
    """
    Remove a known grammatical carrier from a present symptom span
    only when the remaining text exactly matches the symptom name.

    Absent and uncertain symptoms are never modified.
    The original extraction is not mutated.
    """
    payload = extraction.model_dump(mode="python")
    resolved_symptoms = []

    for symptom in payload["symptoms"]:
        reported_text = symptom["reported_text"]

        if symptom["polarity"] != "present":
            resolved_symptoms.append(symptom)
            continue

        resolved_text = reported_text

        # Inspect whitespace boundaries to preserve the exact original
        # substring, including its original Unicode representation.
        for boundary, character in enumerate(reported_text):
            if not character.isspace():
                continue

            prefix = reported_text[:boundary]

            if _comparison_key(prefix) not in _PRESENT_CARRIER_PREFIXES:
                continue

            candidate_start = boundary

            while (
                candidate_start < len(reported_text)
                and reported_text[candidate_start].isspace()
            ):
                candidate_start += 1

            candidate = reported_text[candidate_start:]

            if not candidate:
                continue

            # Conservative condition:
            # the remaining span must match the normalized name exactly.
            if _comparison_key(candidate) != _comparison_key(
                symptom["name"]
            ):
                continue

            # Preserve the source-text invariant.
            if candidate not in user_message:
                continue

            resolved_text = candidate
            break

        symptom["reported_text"] = resolved_text
        resolved_symptoms.append(symptom)

    payload["symptoms"] = resolved_symptoms

    # Rebuild through Pydantic so all model-level validation runs again.
    return SymptomExtractionOutput.model_validate(payload)
