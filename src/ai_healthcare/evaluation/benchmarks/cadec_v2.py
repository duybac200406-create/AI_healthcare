
"""CADECv2 dataset adapter.

Reads original brat-style CADECv2 files without modifying raw text.
The primary evaluation track uses continuous Symptom mentions.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SPLITS = {"train", "dev", "test"}

ENTITY_LABELS = {
    "ADR",
    "Drug",
    "Symptom",
    "Disease",
    "Finding",
}


@dataclass(frozen=True, slots=True)
class TextSegment:
    """One continuous source-text segment."""

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class GoldMention:
    annotation_id: str
    label: str
    segments: tuple[TextSegment, ...]
    annotation_line: int
    annotated_surface: str | None
    offset_status: str = "not_checked"
    @property
    def is_continuous(self) -> bool:
        return len(self.segments) == 1

    @property
    def surface_text(self) -> str | None:
        """Return text only when the mention is continuous."""
        if not self.is_continuous:
            return None
        return self.segments[0].text


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    """One CADECv2 document with its Gold annotations."""

    benchmark_id: str
    split: str
    document_id: str
    language: str
    text: str
    mentions: tuple[GoldMention, ...]
    annotation_notes: tuple[tuple[int, str], ...]
    unclassified_lines: tuple[tuple[int, str], ...]


class CadecV2Adapter:
    """Load and validate CADECv2 brat-style annotations."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

        self.text_dir = self.data_root / "txt"
        self.annotation_dir = self.data_root / "ann"
        self.split_dir = self.data_root / "split"

        for directory in (
            self.text_dir,
            self.annotation_dir,
            self.split_dir,
        ):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Required CADECv2 directory not found: {directory}"
                )

    @staticmethod
    def _read_original_text(path: Path) -> str:
        """Read UTF-8 while preserving original CRLF/LF characters."""
        return path.read_bytes().decode("utf-8")
    @staticmethod
    def _normalize_lf_with_offset_map(
        raw_text: str,
    ) -> tuple[str, tuple[int, ...]]:
        """Normalize newlines and retain LF-to-raw boundary mapping."""

        normalized_chars = []
        raw_boundaries = [0]

        index = 0

        while index < len(raw_text):
            char = raw_text[index]

            if char == "\r":
                normalized_chars.append("\n")

                if (
                    index + 1 < len(raw_text)
                    and raw_text[index + 1] == "\n"
                ):
                    index += 2
                else:
                    index += 1
            else:
                normalized_chars.append(char)
                index += 1

            raw_boundaries.append(index)

        return (
            "".join(normalized_chars),
            tuple(raw_boundaries),
        )
    @staticmethod
    def _classify_offset_status(
        segments: tuple[TextSegment, ...],
        raw_text: str,
        lf_text: str,
        annotated_surface: str | None,
    ) -> str:
        """Classify annotation offsets against raw and LF text."""

        if annotated_surface is None:
            return "missing_surface"

        raw_parts = [
            raw_text[segment.start:segment.end]
            for segment in segments
        ]

        raw_match = (
            " ".join(raw_parts) == annotated_surface
        )

        lf_valid = all(
            0 <= segment.start < segment.end <= len(lf_text)
            for segment in segments
        )

        lf_match = False

        if lf_valid:
            lf_parts = [
                lf_text[segment.start:segment.end]
                for segment in segments
            ]

            lf_match = (
                " ".join(lf_parts) == annotated_surface
            )

        if raw_match and lf_match:
            return "both_match"

        if raw_match:
            return "raw_only"

        if lf_match:
            return "lf_only"

        return "neither"
    def _read_split_ids(self, split: str) -> list[str]:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Expected one of {sorted(SUPPORTED_SPLITS)}."
            )

        split_path = self.split_dir / f"{split}.id"

        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        ids = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        duplicates = [
            item
            for item, count in Counter(ids).items()
            if count > 1
        ]

        if duplicates:
            raise ValueError(
                f"Duplicate document IDs in {split}: "
                f"{duplicates[:10]}"
            )

        return ids

    @staticmethod
    def _parse_span_spec(
        span_spec: str,
        source_text: str,
        annotation_id: str,
    ) -> tuple[TextSegment, ...]:
        segments: list[TextSegment] = []

        for raw_segment in span_spec.split(";"):
            match = re.fullmatch(
                r"\s*(\d+)\s+(\d+)\s*",
                raw_segment,
            )

            if match is None:
                raise ValueError(
                    f"Invalid span syntax for {annotation_id}: "
                    f"{raw_segment!r}"
                )

            start, end = map(int, match.groups())

            if not (0 <= start < end <= len(source_text)):
                raise ValueError(
                    f"Invalid offsets for {annotation_id}: "
                    f"{start}:{end}, text length={len(source_text)}"
                )

            segments.append(
                TextSegment(
                    start=start,
                    end=end,
                    text=source_text[start:end],
                )
            )

        if not segments:
            raise ValueError(
                f"Annotation {annotation_id} has no segments."
            )

        # Preserve source annotation order. Do not silently reorder Gold.
        return tuple(segments)

    @classmethod
    def _parse_annotation_file(
        cls,
        annotation_path: Path,
        source_text: str,
    ) -> tuple[
        tuple[GoldMention, ...],
        tuple[tuple[int, str], ...],
        tuple[tuple[int, str], ...],
    ]:
        mentions: list[GoldMention] = []
        notes: list[tuple[int, str]] = []
        unclassified: list[tuple[int, str]] = []
        seen_ids: set[str] = set()
        lf_text, raw_boundaries = (
            cls._normalize_lf_with_offset_map(source_text)
        )
        lines = annotation_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            # brat annotator notes, e.g.:
            # #1<TAB>AnnotatorNotes T1<TAB>...
            if re.match(r"^#\d+\t", line):
                notes.append((line_number, line))
                continue

            fields = line.split("\t", 2)

            annotation_id = fields[0]

            # The adapter currently consumes text-bound annotations.
            if not re.fullmatch(r"T\d+", annotation_id):
                unclassified.append((line_number, line))
                continue

            if len(fields) < 2:
                raise ValueError(
                    f"Malformed text-bound annotation: "
                    f"{annotation_path.name}:{line_number}"
                )

            descriptor = fields[1].split(maxsplit=1)

            if len(descriptor) != 2:
                raise ValueError(
                    f"Malformed annotation descriptor: "
                    f"{annotation_path.name}:{line_number}"
                )

            label, span_spec = descriptor

            if annotation_id in seen_ids:
                raise ValueError(
                    f"Duplicate annotation ID {annotation_id} "
                    f"in {annotation_path.name}"
                )

            seen_ids.add(annotation_id)

            segments = cls._parse_span_spec(
                span_spec,
                source_text,
                annotation_id,
            )

            annotated_surface = (
                fields[2] if len(fields) == 3 else None
            )

            offset_status = cls._classify_offset_status(
                segments,
                source_text,
                lf_text,
                annotated_surface,
            )
            if offset_status == "lf_only":
                mapped_segments = []

                for segment in segments:
                    if segment.end >= len(raw_boundaries):
                        raise ValueError(
                            "Cannot map LF offset to raw text: "
                            f"{annotation_path.name}:{line_number}"
                        )

                    raw_start = raw_boundaries[segment.start]
                    raw_end = raw_boundaries[segment.end]

                    mapped_segments.append(
                        TextSegment(
                            start=raw_start,
                            end=raw_end,
                            text=source_text[raw_start:raw_end],
                        )
                    )

                segments = tuple(mapped_segments)

            # Preserve the existing Symptom continuous invariant.
            if (
                label == "Symptom"
                and len(segments) == 1
                and offset_status == "neither"
            ):
                raise ValueError(
                    "CADECv2 Symptom surface mismatch: "
                    f"{annotation_path.name}:{line_number}; "
                    f"annotation={annotated_surface!r}; "
                    f"source={segments[0].text!r}"
                )

            mentions.append(
                GoldMention(
                    annotation_id=annotation_id,
                    label=label,
                    segments=segments,
                    annotation_line=line_number,
                    annotated_surface=annotated_surface,
                    offset_status=offset_status,
                )
            )

        return (
            tuple(mentions),
            tuple(notes),
            tuple(unclassified),
        )

    def load_split(
        self,
        split: str,
        *,
        labels: Iterable[str] | None = None,
        include_discontinuous: bool = True,
    ) -> list[BenchmarkDocument]:
        """Load every document in an official split.

        Filtering labels does not remove documents without matching
        mentions. Keeping these documents is important because they
        are negative examples for false-positive evaluation.
        """
        selected_labels = (
            None if labels is None else set(labels)
        )

        if selected_labels is not None:
            unknown = selected_labels - ENTITY_LABELS
            if unknown:
                raise ValueError(
                    f"Unsupported entity labels: {sorted(unknown)}"
                )

        documents: list[BenchmarkDocument] = []

        for document_id in self._read_split_ids(split):
            text_path = self.text_dir / f"{document_id}.txt"
            annotation_path = (
                self.annotation_dir / f"{document_id}.ann"
            )

            if not text_path.is_file():
                raise FileNotFoundError(
                    f"Missing source text: {text_path}"
                )

            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Missing annotation: {annotation_path}"
                )

            source_text = self._read_original_text(text_path)

            mentions, notes, unclassified = (
                self._parse_annotation_file(
                    annotation_path,
                    source_text,
                )
            )

            if selected_labels is not None:
                mentions = tuple(
                    mention
                    for mention in mentions
                    if mention.label in selected_labels
                )

            if not include_discontinuous:
                mentions = tuple(
                    mention
                    for mention in mentions
                    if mention.is_continuous
                )

            documents.append(
                BenchmarkDocument(
                    benchmark_id="cadec_v2",
                    split=split,
                    document_id=document_id,
                    language="en",
                    text=source_text,
                    mentions=mentions,
                    annotation_notes=notes,
                    unclassified_lines=unclassified,
                )
            )

        return documents

    def validate_splits(self) -> dict:
        """Validate official split IDs against available TXT documents."""
        split_sets: dict[str, set[str]] = {}

        for split in sorted(SUPPORTED_SPLITS):
            ids = self._read_split_ids(split)
            split_sets[split] = set(ids)

        all_split_ids = set.union(*split_sets.values())

        all_text_ids = {
            path.stem
            for path in self.text_dir.glob("*.txt")
        }

        all_annotation_ids = {
            path.stem
            for path in self.annotation_dir.glob("*.ann")
        }

        id_occurrences = Counter(
            item
            for ids in split_sets.values()
            for item in ids
        )

        overlaps = {
            item: count
            for item, count in id_occurrences.items()
            if count > 1
        }

        return {
            "split_counts": {
                split: len(ids)
                for split, ids in split_sets.items()
            },
            "total_split_ids": len(all_split_ids),
            "total_text_files": len(all_text_ids),
            "total_annotation_files": len(all_annotation_ids),
            "overlaps": overlaps,
            "missing_text_ids": sorted(
                all_split_ids - all_text_ids
            ),
            "missing_annotation_ids": sorted(
                all_split_ids - all_annotation_ids
            ),
            "unassigned_text_ids": sorted(
                all_text_ids - all_split_ids
            ),
            "valid": (
                not overlaps
                and all_split_ids == all_text_ids
                and all_split_ids == all_annotation_ids
            ),
        }
