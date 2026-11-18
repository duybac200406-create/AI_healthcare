import unicodedata

from hypothesis import given, strategies as st

from ai_healthcare.services.input_guard import (
    fold_diacritics,
    is_patient_record_request,
    normalize_user_text,
)

# Messages copied from the existing Commander guard tests
# (tests/unit/test_commander.py) so behaviour stays at parity.
BLOCKED = (
    "Tra cứu hồ sơ bệnh án của bệnh nhân.",
    "Đọc hồ sơ cá nhân của tôi.",
    "Cho tôi biết thông tin trong hồ sơ SYN-001.",
)
NOT_BLOCKED = ("Những thông tin nào thường được ghi trong hồ sơ bệnh án?",)


@given(st.text())
def test_normalization_is_independent_of_unicode_form(text):
    nfc = normalize_user_text(unicodedata.normalize("NFC", text))
    nfd = normalize_user_text(unicodedata.normalize("NFD", text))
    assert nfc == nfd


@given(st.text())
def test_normalization_is_idempotent(text):
    once = normalize_user_text(text)
    assert normalize_user_text(once) == once


@given(st.text())
def test_fold_diacritics_leaves_no_combining_marks(text):
    folded = fold_diacritics(text)
    assert all(unicodedata.category(c) != "Mn" for c in folded)
    assert fold_diacritics(folded) == folded


def test_guard_matches_existing_cases_in_every_text_form():
    for message in BLOCKED:
        for variant in (
            message,
            unicodedata.normalize("NFD", message),
            fold_diacritics(message),
            message.upper(),
        ):
            assert is_patient_record_request(variant), variant
    for message in NOT_BLOCKED:
        for variant in (
            message,
            unicodedata.normalize("NFD", message),
            fold_diacritics(message),
        ):
            assert not is_patient_record_request(variant), variant
