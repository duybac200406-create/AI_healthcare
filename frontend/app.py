"""Streamlit chat demo: Commander -> Symptom -> Medical RAG.

All orchestration lives in ConversationService; this file only renders.
Conversation state is held in st.session_state for the current browser
session (persistence is a later phase).
"""

import uuid

import streamlit as st

from ai_healthcare.schemas.conversation import ConversationState, TurnResult
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.services import runtime as runtime_module

st.set_page_config(
    page_title="AI Healthcare — Chat Demo",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_runtime():
    """Build models and agents once per server process."""

    return runtime_module.build_runtime(runtime_module.RuntimeConfig.from_env())


def new_conversation() -> None:
    st.session_state.state = ConversationState.new(uuid.uuid4().hex)
    st.session_state.transcript = []


if "state" not in st.session_state:
    new_conversation()


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_answer(result: TurnResult) -> None:
    output: MedicalRAGOutput = result.medical_output
    titles = {e.evidence_id: e.title for e in output.retrieved_evidence}

    if output.status == "answered":
        for claim in output.claims:
            st.markdown(f"- {claim.text}")
            sources = ", ".join(
                f"{titles.get(eid, eid)} ({eid})" for eid in claim.evidence_ids
            )
            st.caption(f"Nguồn: {sources}")
    else:
        st.warning(output.answer_text)

    if result.coverage_notice:
        st.caption(result.coverage_notice)
    if result.language_notice:
        st.caption(result.language_notice)


def render_details(result: TurnResult) -> None:
    with st.expander("Chi tiết kỹ thuật"):
        st.write(f"Kết quả lượt: `{result.kind}` · {result.latency_ms} ms")

        if result.proposed_plan is not None:
            proposed = [t.agent for t in result.proposed_plan.tasks]
            st.write(
                f"Commander: intent `{result.proposed_plan.intent}`, "
                f"agents `{proposed}`"
            )
        if result.executed_plan is not None:
            executed = [t.agent for t in result.executed_plan.tasks]
            st.write(f"Plan đã thực thi: `{executed}`")

        if result.gate is not None:
            st.write(
                f"Cổng đủ thông tin: allow=`{result.gate.allow}`, "
                f"lý do `{result.gate.reason_code}`, "
                f"giới hạn thông tin=`{result.gate.limited_information}`"
            )

        if result.symptom_output is not None:
            for m in result.symptom_output.symptoms:
                st.write(
                    f"Triệu chứng: **{m.name}** (`{m.polarity}`) · "
                    f"khởi phát={m.onset_text!r} · kéo dài={m.duration_text!r} "
                    f"· mức độ={m.severity_text!r}"
                )

        output = result.medical_output
        if output is not None:
            st.write(
                f"Medical RAG: `{output.status}` · mô hình "
                f"`{output.generation_model}`"
            )
            if result.language_outcome:
                st.write(f"Kiểm tra ngôn ngữ: `{result.language_outcome}`")
            for e in output.retrieved_evidence:
                st.markdown(
                    f"`{e.evidence_id}` · **{e.title}** · "
                    f"similarity `{e.similarity_score:.3f}` · "
                    f"`{e.document_id}#{e.chunk_index}`"
                )
                quoted = e.excerpt.replace("\n", "\n> ")
                st.markdown(f"> {quoted}")
                if e.source_url:
                    st.markdown(f"[Mở nguồn tài liệu]({e.source_url})")


def render_result(result: TurnResult) -> None:
    if result.kind == "clarification":
        st.markdown("Để hiểu rõ hơn, bạn cho mình biết thêm:")
        for question in result.questions:
            st.markdown(f"- {question}")
    elif result.kind == "answer" and result.medical_output is not None:
        render_answer(result)
    elif result.kind == "unsupported":
        st.info(result.message)
    else:
        st.warning(result.message or "Không thể xử lý yêu cầu.")

    render_details(result)


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

st.title("AI Healthcare — Chat Demo")
st.warning(
    "Research prototype only. Không sử dụng thông tin sức khỏe thật. "
    "Nội dung chỉ mang tính tham khảo, không phải chẩn đoán hay hướng dẫn "
    "điều trị."
)

config = runtime_module.RuntimeConfig.from_env()
with st.sidebar:
    st.markdown("### Cấu hình")
    st.write(f"LLM: `{config.chat_model}`")
    st.write(f"Embedding: `{config.embedding_model}`")
    st.write(f"Qdrant: `{config.qdrant_url}`")
    st.write(f"Collection: `{config.collection}`")
    st.caption("Demo bật Symptom Agent và Medical RAG Agent.")
    if st.button("Cuộc trò chuyện mới", use_container_width=True):
        new_conversation()
        st.rerun()

if not st.session_state.transcript:
    st.info(
        "Bạn có thể mô tả triệu chứng hoặc đặt một câu hỏi y khoa chung."
    )

for entry in st.session_state.transcript:
    with st.chat_message(entry["role"]):
        if entry["role"] == "user":
            st.markdown(entry["text"])
        else:
            render_result(entry["result"])

prompt = st.chat_input("Nhập câu hỏi hoặc mô tả triệu chứng của bạn...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang xử lý..."):
                result = get_runtime().service.handle_turn(
                    st.session_state.state, prompt
                )
        except Exception as exc:
            st.error(
                f"Không thể chạy hệ thống: {type(exc).__name__}: {exc}"
            )
        else:
            st.session_state.state = result.state
            st.session_state.transcript.append(
                {"role": "user", "text": prompt}
            )
            st.session_state.transcript.append(
                {"role": "assistant", "result": result}
            )
            render_result(result)
