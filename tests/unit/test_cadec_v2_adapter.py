
from pathlib import Path

import pytest

from ai_healthcare.evaluation.benchmarks.cadec_v2 import (
    CadecV2Adapter,
)


def create_dataset(
    root: Path,
    *,
    text: str,
    annotations: str,
    document_id: str = "DOC001",
    split: str = "dev",
) -> Path:
    data = root / "data"

    for folder in ("txt", "ann", "split"):
        (data / folder).mkdir(parents=True, exist_ok=True)

    # Write source text as bytes to preserve CRLF exactly.
    (data / "txt" / f"{document_id}.txt").write_bytes(
        text.encode("utf-8")
    )

    (data / "ann" / f"{document_id}.ann").write_text(
        annotations,
        encoding="utf-8",
    )

    (data / "split" / f"{split}.id").write_text(
        f"{document_id}\n",
        encoding="utf-8",
    )

    for other_split in {"train", "dev", "test"} - {split}:
        path = data / "split" / f"{other_split}.id"
        path.write_text("", encoding="utf-8")

    return data


def test_preserves_original_crlf_and_character_offsets(tmp_path):
    text = "I have pain\r\nand nausea."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tSymptom 7 11\tpain\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split(
        "dev",
        labels={"Symptom"},
        include_discontinuous=False,
    )

    assert len(documents) == 1
    assert documents[0].text == text
    assert documents[0].mentions[0].surface_text == "pain"
    assert documents[0].mentions[0].segments[0].start == 7


def test_preserves_discontinuous_segments(tmp_path):
    text = "I have pain and fever."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tSymptom 7 11;16 21\tpain fever\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev")

    mention = documents[0].mentions[0]

    assert mention.is_continuous is False
    assert [
        segment.text for segment in mention.segments
    ] == ["pain", "fever"]


def test_label_filter_keeps_documents_without_matching_mentions(
    tmp_path,
):
    text = "The patient takes aspirin."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tDrug 17 24\taspirin\n",
    )

    adapter = CadecV2Adapter(data_root)

    documents = adapter.load_split(
        "dev",
        labels={"Symptom"},
    )

    assert len(documents) == 1
    assert documents[0].mentions == ()


def test_preserves_annotator_notes_and_unclassified_lines(
    tmp_path,
):
    text = "I have pain."

    annotations = (
        "T1\tSymptom 7 11\tpain\n"
        "#1\tAnnotatorNotes T1\tExample note\n"
        "unclassified continuation\n"
    )

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations=annotations,
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev")

    assert len(documents[0].annotation_notes) == 1
    assert len(documents[0].unclassified_lines) == 1


def test_rejects_invalid_offsets(tmp_path):
    text = "I have pain."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tSymptom 7 99\tpain\n",
    )

    adapter = CadecV2Adapter(data_root)

    with pytest.raises(ValueError, match="Invalid offsets"):
        adapter.load_split("dev")


def test_validates_official_split_membership(tmp_path):
    data_root = create_dataset(
        tmp_path,
        text="I have pain.",
        annotations="T1\tSymptom 7 11\tpain\n",
    )

    adapter = CadecV2Adapter(data_root)
    result = adapter.validate_splits()

    assert result["valid"] is True
    assert result["split_counts"]["dev"] == 1


def test_rejects_duplicate_ids_within_split(tmp_path):
    data_root = create_dataset(
        tmp_path,
        text="I have pain.",
        annotations="T1\tSymptom 7 11\tpain\n",
    )

    split_path = data_root / "split" / "dev.id"
    split_path.write_text(
        "DOC001\nDOC001\n",
        encoding="utf-8",
    )

    adapter = CadecV2Adapter(data_root)

    with pytest.raises(ValueError, match="Duplicate document IDs"):
        adapter.load_split("dev")

def test_remaps_lf_only_offsets_to_raw_coordinates(tmp_path):
    text = "A\r\nB pain."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tSymptom 4 8\tpain\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev", labels={"Symptom"})

    mention = documents[0].mentions[0]
    segment = mention.segments[0]

    assert mention.annotated_surface == "pain"
    assert mention.offset_status == "lf_only"

    # LF offsets [4, 8) must map to raw offsets [5, 9).
    assert (segment.start, segment.end) == (5, 9)
    assert segment.text == "pain"

    # Original document text must remain unchanged.
    assert documents[0].text == text


def test_preserves_raw_only_offsets(tmp_path):
    text = "A\r\nB pain."

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tADR 5 9\tpain\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev", labels={"ADR"})

    mention = documents[0].mentions[0]
    segment = mention.segments[0]

    assert mention.annotated_surface == "pain"
    assert mention.offset_status == "raw_only"

    assert (segment.start, segment.end) == (5, 9)
    assert segment.text == "pain"


def test_preserves_neither_annotation_without_silent_correction(
    tmp_path,
):
    text = "bad heart\nPalpitations"

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tADR 0 22\tbad heart\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev", labels={"ADR"})

    mention = documents[0].mentions[0]
    segment = mention.segments[0]

    assert mention.annotated_surface == "bad heart"
    assert mention.offset_status == "neither"

    # Do not silently shorten the original annotation.
    assert (segment.start, segment.end) == (0, 22)
    assert segment.text == "bad heart\nPalpitations"


def test_preserves_reversed_discontinuous_segment_order(tmp_path):
    text = "joint and pain"

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations="T1\tSymptom 10 14;0 5\tpain joint\n",
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev")

    mention = documents[0].mentions[0]

    assert mention.is_continuous is False
    assert mention.annotated_surface == "pain joint"
    assert mention.offset_status == "both_match"

    assert [
        (segment.start, segment.end)
        for segment in mention.segments
    ] == [
        (10, 14),
        (0, 5),
    ]

    assert [
        segment.text for segment in mention.segments
    ] == ["pain", "joint"]


def test_preserves_discontinuous_neither_annotation(tmp_path):
    text = "felt like severe food\npoisoning"

    data_root = create_dataset(
        tmp_path,
        text=text,
        annotations=(
            "T1\tADR 0 9;10 31\tfelt like severe food\n"
        ),
    )

    adapter = CadecV2Adapter(data_root)
    documents = adapter.load_split("dev", labels={"ADR"})

    mention = documents[0].mentions[0]

    assert mention.annotated_surface == "felt like severe food"
    assert mention.offset_status == "neither"

    assert [
        (segment.start, segment.end)
        for segment in mention.segments
    ] == [
        (0, 9),
        (10, 31),
    ]

    assert [
        segment.text for segment in mention.segments
    ] == [
        "felt like",
        "severe food\npoisoning",
    ]
