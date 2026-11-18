"""Closed-set detection of social messages (greeting, thanks, farewell, ack).

Deliberately conservative: a message counts as social only when EVERY token is
part of a small social vocabulary. Anything containing other words (for
example "dạ có" or "cảm ơn, tôi bị đau") is not social and goes through the
normal pipeline. This is conversation management, not medical classification.
"""

from __future__ import annotations

import re
from typing import Literal

from ai_healthcare.services.input_guard import fold_diacritics

SocialKind = Literal["greeting", "thanks", "farewell", "ack"]

_PUNCT_RE = re.compile(r"[^\w\s]|_")
_MAX_TOKENS = 6

_FILLERS = frozenset(
    {"ban", "nhe", "nha", "oi", "a", "minh", "em", "anh", "chi",
     "nhieu", "qua", "rat", "you", "very", "much", "so", "there"}
)
_VOCAB: dict[SocialKind, frozenset[str]] = {
    "greeting": frozenset({"xin", "chao", "hello", "hi", "hey", "alo"}),
    "thanks": frozenset({"cam", "on", "thanks", "thank", "thankyou"}),
    "farewell": frozenset({"tam", "biet", "bye", "goodbye", "hen", "gap", "lai"}),
    "ack": frozenset({"ok", "oke", "okay", "vang", "da", "u", "uh", "um", "ah"}),
}
_TRIGGERS: dict[SocialKind, tuple[tuple[str, ...], ...]] = {
    "greeting": (("chao",), ("hello",), ("hi",), ("hey",), ("alo",)),
    "thanks": (("cam", "on"), ("thanks",), ("thank", "you"), ("thankyou",)),
    "farewell": (("tam", "biet"), ("bye",), ("goodbye",), ("hen", "gap", "lai")),
    "ack": (("ok",), ("oke",), ("okay",), ("vang",), ("da",), ("u",),
            ("uh",), ("um",), ("ah",)),
}

REPLIES: dict[SocialKind, str] = {
    "greeting": (
        "Xin chào! Bạn có thể mô tả triệu chứng đang gặp hoặc đặt một câu "
        "hỏi y khoa chung."
    ),
    "thanks": (
        "Rất vui được hỗ trợ. Nội dung chỉ mang tính tham khảo và không "
        "thay thế tư vấn của nhân viên y tế."
    ),
    "farewell": "Chào bạn, chúc bạn nhiều sức khỏe.",
    "ack": (
        "Bạn có thể mô tả triệu chứng đang gặp hoặc đặt một câu hỏi y khoa "
        "chung."
    ),
}


def _contains(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1))


def classify_social(text: str) -> SocialKind | None:
    folded = _PUNCT_RE.sub(" ", fold_diacritics(text))
    tokens = folded.split()
    if not tokens or len(tokens) > _MAX_TOKENS:
        return None

    for kind in ("greeting", "thanks", "farewell", "ack"):
        vocabulary = _VOCAB[kind] | _FILLERS
        if all(token in vocabulary for token in tokens) and any(
            _contains(tokens, phrase) for phrase in _TRIGGERS[kind]
        ):
            return kind
    return None
