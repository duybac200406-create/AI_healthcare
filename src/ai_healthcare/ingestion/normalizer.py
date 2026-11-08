
"""Deterministic text normalization for the ingestion pipeline.

Normalization is a separate pipeline step:

read -> normalize -> chunk

Character offsets always refer to the exact normalized string stored
in MedicalDocument.content.

Rules:
1. Remove consecutive U+FEFF characters at the beginning only.
2. Convert CRLF and lone CR to LF.
3. Normalize Unicode to NFC.

Zero-width characters, including U+200B, U+200C and U+200D,
are preserved. U+FEFF occurring inside the text is also preserved.

Whitespace collapsing, case folding and diacritic removal are
deliberately not performed.

Any change to these rules MUST bump NORMALIZER_VERSION.
"""

import unicodedata

from pydantic import ValidationError

from ai_healthcare.schemas.document import MedicalDocument


__all__ = [
    "NORMALIZER_VERSION",
    "NormalizationError",
    "normalize_document",
    "normalize_text",
]

NORMALIZER_VERSION = "normalizer-v1"

_BOM = "\ufeff"


class NormalizationError(ValueError):
    """Raised when a document cannot be normalized into valid content.

    Error messages never include document content.
    """


def normalize_text(text: str) -> str:
    """Return text normalized according to the module rules.

    The function is deterministic and idempotent:

        normalize_text(normalize_text(x)) == normalize_text(x)

    Zero-width characters are preserved.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a str.")

    # Remove BOM only when it occurs at the beginning.
    # lstrip removes consecutive leading BOM characters.
    result = text.lstrip(_BOM)

    # Normalize line endings.
    result = result.replace("\r\n", "\n").replace("\r", "\n")

    # Unicode NFC.
    return unicodedata.normalize("NFC", result)


def normalize_document(document: MedicalDocument) -> MedicalDocument:

    if not isinstance(document, MedicalDocument):
        raise TypeError(
            "document must be a MedicalDocument instance."
        )

    normalized = normalize_text(document.content)

    has_usable_content = any(
        not character.isspace()
        and unicodedata.category(character) != "Cf"
        for character in normalized
    )

    if not has_usable_content:
        raise NormalizationError(
            "Document content is empty after normalization."
        )

    data = document.model_dump(mode="json")
    data["content"] = normalized

    try:
        return MedicalDocument.model_validate(data)

    except ValidationError:
        raise NormalizationError(
            "Normalized document failed MedicalDocument validation."
        ) from None
