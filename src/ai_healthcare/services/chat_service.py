"""Deterministic, evidence-grounded response logic for MVP v0.1.

This intentionally does not call an LLM. It establishes testable source-of-truth
and uncertainty behavior before adding a generative model.
"""

from collections.abc import Mapping

from ai_healthcare.schemas.patient import SyntheticPatientRecord


def answer_question(
    question: str,
    patient: SyntheticPatientRecord,
    evidence: Mapping[str, str],
) -> str:
    """Answer the supported synthetic latex-status question, otherwise abstain."""

    normalized_question = question.strip().casefold()
    if not normalized_question:
        return "Vui lòng nhập câu hỏi."

    latex_terms = ("latex", "dị ứng", "di ung", "allergy")
    if not any(term in normalized_question for term in latex_terms):
        return (
            "Phiên bản MVP hiện chỉ hỗ trợ minh họa câu hỏi về trạng thái dị ứng latex. "
            "Tôi chưa đủ chức năng để trả lời câu hỏi này."
        )

    if patient.allergy_status == "RECORDED_AS_CONFIRMED":
        status_text = (
            "Hồ sơ ghi trạng thái dị ứng latex là `RECORDED AS CONFIRMED`. "
            "Đây là trạng thái được ghi trong hồ sơ; dữ liệu hiện có không mô tả "
            "cơ sở xác nhận, mức độ nghiêm trọng, triệu chứng hoặc phản ứng trước đây."
        )
    else:  # NOT_ASSESSED
        status_text = (
            "Hồ sơ ghi trạng thái dị ứng latex là `NOT ASSESSED` (CHƯA ĐƯỢC ĐÁNH GIÁ). "
            "Không thể kết luận từ hồ sơ này rằng bệnh nhân có hoặc không có dị ứng latex. "
            "Mức độ nghiêm trọng, triệu chứng và phản ứng trước đây chưa được cung cấp."
        )

    e1 = evidence.get("E1")
    e2 = evidence.get("E2")
    if not e1 or not e2:
        evidence_text = (
            "Chưa có đủ các trích đoạn bằng chứng bắt buộc để đưa ra phần hướng dẫn chung; "
            "tôi không tự bổ sung kiến thức bên ngoài."
        )
    else:
        evidence_text = (
            f"**Thông tin chung từ bằng chứng**\n\n"
            f"- [E1] {e1}\n\n"
            f"- [E2] {e2}\n\n"
            "Đây là thông tin chung, không chứng minh các khám, xét nghiệm hoặc biện pháp "
            "tránh latex đã được thực hiện đối với hồ sơ này."
        )

    return (
        f"### Trạng thái ghi nhận trong hồ sơ\n\n"
        f"{status_text} [SYN-001]\n\n"
        f"### {evidence_text}\n\n"
        f"### Giới hạn\n\n"
        "Không suy diễn thêm về chẩn đoán, kết quả xét nghiệm, điều trị hay hành vi tránh latex. "
        "Đây là bản demo dùng dữ liệu giả lập, không phải tư vấn y khoa."
    )
