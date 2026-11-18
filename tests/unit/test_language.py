import unicodedata

from ai_healthcare.services.language import detect_language

VI = "Đây là một câu trả lời được viết bằng tiếng Việt."
EN = "This is a sentence written in plain English words."


def test_vietnamese_is_detected_in_either_unicode_form():
    assert detect_language(VI) == "vi"
    assert detect_language(unicodedata.normalize("NFD", VI)) == "vi"


def test_english_is_other():
    assert detect_language(EN) == "other"


def test_short_text_is_unknown():
    assert detect_language("ok") == "unknown"
    assert detect_language("") == "unknown"


def test_unaccented_vietnamese_is_not_claimed_as_vietnamese():
    # Documented limitation: callers must treat "other" as "not known to be vi".
    assert detect_language("toi bi dau o cho nay tu hom qua") == "other"
