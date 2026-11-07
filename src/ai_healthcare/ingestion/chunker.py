import hashlib

from ai_healthcare.schemas.document import (
    DocumentChunk,
    MedicalDocument,
)


def chunk_document(
    document: MedicalDocument,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> tuple[DocumentChunk, ...]:

    if not isinstance(document, MedicalDocument):
        raise TypeError("document must be a MedicalDocument instance.")

    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer.")

    if isinstance(overlap_chars, bool) or not isinstance(
        overlap_chars, int
    ):
        raise TypeError("overlap_chars must be an integer.")

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")

    if overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be smaller than max_chars."
        )

    source = document.content
    document_hash = hashlib.sha256(
        document.document_id.encode("utf-8")
    ).hexdigest()[:12]

    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(source):

        # Skip whitespace before starting a new chunk.
        while start < len(source) and source[start].isspace():
            start += 1

        if start >= len(source):
            break

        hard_end = min(start + max_chars, len(source))
        end = hard_end

        # Prefer a whitespace boundary when possible.
        if hard_end < len(source):
            boundary = hard_end

            while boundary > start and not source[boundary - 1].isspace():
                boundary -= 1

            if boundary > start:
                end = boundary

        # Exclude trailing whitespace from the chunk.
        while end > start and source[end - 1].isspace():
            end -= 1

        if end <= start:
            end = hard_end

        chunk_content = source[start:end]

        chunk = DocumentChunk(
            chunk_id=f"chunk_{document_hash}_{len(chunks):04d}",
            document_id=document.document_id,
            chunk_index=len(chunks),
            content=chunk_content,
            language=document.language,
            start_char=start,
            end_char=end,
        )

        chunks.append(chunk)

        if end >= len(source):
            break

        # Move backward to create overlap, then advance to a word
        # boundary so that a word is not split between chunks.
        next_start = max(start + 1, end - overlap_chars)

        while next_start < end and not source[next_start].isspace():
            next_start += 1

        while next_start < len(source) and source[next_start].isspace():
            next_start += 1

        if next_start <= start:
            next_start = end

        start = next_start

    return tuple(chunks)
