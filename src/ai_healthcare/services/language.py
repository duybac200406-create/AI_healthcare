"""Deterministic, dependency-free language heuristic for Vietnamese vs other.

It only answers one question: "is this text written in Vietnamese?". It looks
at the share of Latin letters that carry Vietnamese diacritics. Text without
diacritics (for example unaccented Vietnamese) is reported as "other", so any
caller must treat "other" as "not known to be Vietnamese".
"""

from __future__ import annotations

import unicodedata
from typing import Literal

Language = Literal["vi", "other", "unknown"]

MIN_LETTERS = 8
VIETNAMESE_RATIO = 0.05


def _is_marked_letter(ch: str) -> bool:
    if ch in "đĐ":
        return True
    decomposed = unicodedata.normalize("NFD", ch)
    return len(decomposed) > 1 and any(
        unicodedata.category(c) == "Mn" for c in decomposed[1:]
    )


def detect_language(text: str) -> Language:
    normalized = unicodedata.normalize("NFC", text)
    letters = [c for c in normalized if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return "unknown"
    marked = sum(1 for c in letters if _is_marked_letter(c))
    return "vi" if marked / len(letters) >= VIETNAMESE_RATIO else "other"
