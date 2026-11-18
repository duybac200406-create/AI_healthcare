"""Input normalization and accent-insensitive pre-filters for user messages.

Why this module exists
----------------------
Vietnamese text can reach the pipeline in different Unicode forms (NFC/NFD)
or without diacritics. Downstream components compare strings exactly (for
example the Symptom Agent's source-span check) or use diacritic-bearing
regular expressions (the Commander's patient-record guard). Normalizing once,
at the single entry point, keeps those components unchanged.
"""

from __future__ import annotations

import re
import unicodedata

MAX_USER_MESSAGE_CHARS = 4000

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_user_text(text: str) -> str:
    """Return NFC-normalized text with collapsed whitespace.

    Control characters are removed. The result is what every agent sees.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    normalized = unicodedata.normalize("NFC", text)
    normalized = "".join(
        ch for ch in normalized
        if ch in "\n\t " or unicodedata.category(ch)[0] != "C"
    )
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def fold_diacritics(text: str) -> str:
    """Casefold and strip diacritics (Vietnamese đ/Đ included)."""

    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    return stripped.replace("đ", "d")


# Folded counterparts of Commander._is_patient_record_lookup. They are applied
# to fold_diacritics(message), so they contain no diacritics.
_RECORD_NOUN = r"(?:ho so|benh an|medical record|health record|patient record)"
_OWNER = (
    r"(?:cua toi|cua minh|ca nhan|benh nhan|my |patient|syn-\d+|"
    r"of patient)"
)
_ACTION = (
    r"(?:tra cuu|truy xuat|truy cap|doc|xem|mo|lay|kiem tra|cho toi|"
    r"show|open|read|access|retrieve|look up|view)"
)

_FOLDED_PATIENT_RECORD_PATTERNS = (
    re.compile(rf"\b{_ACTION}\b.{{0,80}}\b{_RECORD_NOUN}\b.{{0,80}}{_OWNER}"),
    re.compile(rf"\b{_RECORD_NOUN}\b.{{0,40}}\b(?:syn-\d+)\b"),
    re.compile(rf"{_OWNER}.{{0,40}}\b{_RECORD_NOUN}\b"),
)

# Educational wording that describes records in general is not a lookup.
_GENERIC_RECORD_QUESTION = re.compile(
    r"\b(?:nhung thong tin nao|thuong duoc ghi|la gi|what is|what are)\b"
)


def is_patient_record_request(message: str) -> bool:
    """Accent-insensitive pre-filter for individual patient-record lookups.

    This is an additional layer in front of the Commander's own guard; it
    never replaces it. It is deliberately conservative: educational questions
    about what a record contains are not blocked.
    """

    folded = fold_diacritics(normalize_user_text(message))
    if _GENERIC_RECORD_QUESTION.search(folded) and "syn-" not in folded:
        return False
    return any(p.search(folded) for p in _FOLDED_PATIENT_RECORD_PATTERNS)
