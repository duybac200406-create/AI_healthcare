"""Safety-oriented unit tests for the synthetic allergy-status demo."""

import json
from pathlib import Path

from ai_healthcare.schemas.patient import SyntheticPatientRecord
from ai_healthcare.services.chat_service import answer_question

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "data" / "synthetic" / "evidence" / "latex_allergy.json"


def load_evidence() -> dict[str, str]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def make_patient(status: str) -> SyntheticPatientRecord:
    return SyntheticPatientRecord(
        patient_id="SYN-001",
        display_name="Bệnh nhân giả lập 001",
        allergy_name="latex",
        allergy_status=status,
        severity=None,
        symptoms=[],
        previous_reactions=[],
        notes="Dữ liệu giả lập dùng để kiểm thử.",
    )


def test_not_assessed_is_not_misrepresented_as_present_or_absent() -> None:
    answer = answer_question(
        "Tình trạng dị ứng latex là gì?",
        make_patient("NOT_ASSESSED"),
        load_evidence(),
    )
    assert "NOT ASSESSED" in answer
    assert "Không thể kết luận" in answer
    assert "[SYN-001]" in answer
    assert "[E1]" in answer
    assert "[E2]" in answer


def test_recorded_as_confirmed_status_is_preserved() -> None:
    answer = answer_question(
        "Tôi có dị ứng latex không?",
        make_patient("RECORDED_AS_CONFIRMED"),
        load_evidence(),
    )
    assert "RECORDED AS CONFIRMED" in answer
    assert "cơ sở xác nhận" in answer
    assert "[SYN-001]" in answer


def test_unsupported_question_abstains() -> None:
    answer = answer_question(
        "Tôi nên uống thuốc nào?",
        make_patient("NOT_ASSESSED"),
        load_evidence(),
    )
    assert "chưa đủ chức năng" in answer
