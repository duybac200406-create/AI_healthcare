"""Streamlit UI for AI Healthcare MVP v0.1."""

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from ai_healthcare.schemas.patient import SyntheticPatientRecord
from ai_healthcare.services.chat_service import answer_question

ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = ROOT / "data" / "synthetic" / "patients" / "SYN-001.json"
EVIDENCE_FILE = ROOT / "data" / "synthetic" / "evidence" / "latex_allergy.json"

st.set_page_config(page_title="AI Healthcare — MVP", page_icon="🩺", layout="centered")
st.title("AI Healthcare — MVP v0.1")
st.caption("Bản thử nghiệm phần mềm · Dữ liệu bệnh nhân hoàn toàn giả lập")
st.warning(
    "Không nhập thông tin sức khỏe thật. Bản demo chưa phải công cụ chẩn đoán "
    "hoặc tư vấn điều trị và chưa được xác nhận an toàn lâm sàng."
)

@st.cache_data
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

try:
    patient_data = load_json(PATIENT_FILE)
    evidence = load_json(EVIDENCE_FILE)
    patient = SyntheticPatientRecord.model_validate(patient_data)
except (OSError, json.JSONDecodeError, ValidationError) as exc:
    st.error(f"Không thể tải hoặc xác thực dữ liệu demo: {exc}")
    st.stop()

with st.expander("Xem hồ sơ giả lập đang dùng"):
    st.write(f"**Mã hồ sơ:** {patient.patient_id}")
    st.write(f"**Tên hiển thị:** {patient.display_name}")
    st.write(f"**Trường dị ứng:** {patient.allergy_name}")
    st.write(f"**Trạng thái ghi nhận:** `{patient.allergy_status}`")
    st.write("**Mức độ:** Chưa có dữ liệu" if patient.severity is None else f"**Mức độ:** {patient.severity}")

with st.expander("Xem bằng chứng cố định"):
    for evidence_id, excerpt in evidence.items():
        st.markdown(f"**[{evidence_id}]** {excerpt}")

st.subheader("Đặt câu hỏi")
question = st.text_area(
    "Nhập câu hỏi bằng tiếng Việt",
    placeholder="Ví dụ: Trạng thái dị ứng latex trong hồ sơ của tôi là gì?",
    height=100,
)

if st.button("Tạo câu trả lời", type="primary", use_container_width=True):
    if not question.strip():
        st.info("Vui lòng nhập câu hỏi trước khi gửi.")
    else:
        answer = answer_question(question, patient, evidence)
        st.subheader("Câu trả lời")
        st.markdown(answer)

st.divider()
st.caption(
    "Giới hạn phiên bản: chỉ minh họa một loại câu hỏi và dữ liệu cố định; "
    "chưa tích hợp LLM, Medical RAG, tài khoản, hoặc lưu trữ dữ liệu người dùng."
)
