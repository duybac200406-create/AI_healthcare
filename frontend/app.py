
"""Streamlit demo: Commander -> Symptom -> Medical RAG."""

import os
import time
import uuid

import streamlit as st
from qdrant_client import QdrantClient

from ai_healthcare.agents.commander import Commander, CommanderPolicyError
from ai_healthcare.agents.symptom import SymptomAgent
from ai_healthcare.agents.medical_rag import MedicalRAGAgent
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.orchestration.response_assembler import ResponseAssembler
from ai_healthcare.providers.ollama_provider import OllamaProvider
from ai_healthcare.providers.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.agent import AgentTask, CommanderPlan
from ai_healthcare.schemas.execution import AgentExecutionResult
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.schemas.symptom import SymptomExtractionOutput
from ai_healthcare.schemas.vector import VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OLLAMA_URL = os.getenv(
    "AI_HEALTHCARE_OLLAMA_URL",
    "http://127.0.0.1:11434",
)
OLLAMA_MODEL = os.getenv(
    "AI_HEALTHCARE_CHAT_MODEL",
    "qwen3:8b",
)
EMBEDDING_MODEL = "bge-m3"

QDRANT_URL = os.getenv(
    "AI_HEALTHCARE_QDRANT_URL",
    "http://127.0.0.1:6333",
)
COLLECTION_NAME = os.getenv(
    "AI_HEALTHCARE_QDRANT_COLLECTION",
    "medical_knowledge_bge_m3_multidomain_plaintext_v1",
)


# ---------------------------------------------------------------------
# Runtime initialization — cached across Streamlit reruns
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_runtime():
    """Initialize shared local models, existing Qdrant collection and agents."""

    # IMPORTANT:
    # Check the existing collection before calling ensure_collection().
    # QdrantVectorStore.ensure_collection() can create a missing collection.
    probe = QdrantClient(url=QDRANT_URL, timeout=10.0)

    try:
        if not probe.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Refusing to create an empty collection automatically."
            )

        point_count = probe.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        if point_count <= 0:
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' is empty."
            )
    finally:
        probe.close()

    llm_provider = OllamaProvider(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
    )

    embedding_provider = OllamaEmbeddingProvider(
        base_url=OLLAMA_URL,
        model=EMBEDDING_MODEL,
    )

    vector_store = QdrantVectorStore(
        url=QDRANT_URL,
        config=VectorStoreConfig(
            collection_name=COLLECTION_NAME,
            vector_size=1024,
            distance="cosine",
            embedding_model=EMBEDDING_MODEL,
        ),
    )

    try:
        # Validate the existing collection's vector configuration.
        vector_store.ensure_collection()

        retriever = DenseRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        symptom_agent = SymptomAgent(
            provider=llm_provider,
        )

        medical_rag_agent = MedicalRAGAgent(
            provider=llm_provider,
            retriever=retriever,
            top_k=5,
        )

        registry = AgentRegistry()
        registry.register(symptom_agent)
        registry.register(medical_rag_agent)

        # Only these two agents are enabled in demo v0.1.
        orchestrator = Orchestrator(
            enabled_agents={"symptom", "medical_rag"},
            registry=registry,
        )

        commander = Commander(
            provider=llm_provider,
            orchestrator=orchestrator,
        )

        assembler = ResponseAssembler()

        return {
            "commander": commander,
            "orchestrator": orchestrator,
            "assembler": assembler,
            "llm_model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "collection": COLLECTION_NAME,
            "point_count": point_count,
        }

    except Exception:
        vector_store.close()
        embedding_provider.close()
        llm_provider.close()
        raise


# ---------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------

def render_symptom_output(output: SymptomExtractionOutput):
    st.markdown("### Symptom extraction")

    st.write(f"Trạng thái: `{output.status}`")

    if not output.symptoms:
        st.info("Không có mention triệu chứng được trích xuất.")
    else:
        for index, mention in enumerate(output.symptoms, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {mention.name}**")
                st.write(f"Văn bản gốc: `{mention.reported_text}`")
                st.write(f"Polarity: `{mention.polarity}`")

                details = {
                    "Thời điểm khởi phát": mention.onset_text,
                    "Thời gian kéo dài": mention.duration_text,
                    "Mức độ": mention.severity_text,
                    "Vị trí": mention.body_site_text,
                }

                for label, value in details.items():
                    if value is not None:
                        st.write(f"{label}: {value}")

    if output.follow_up_questions:
        st.markdown("**Thông tin cần làm rõ:**")
        for question in output.follow_up_questions:
            st.write(f"- {question}")

    st.info(
        "Giới hạn hiện tại: schema Symptom chưa phân loại riêng ADR. "
        "Không được diễn giải các mention này thành nhãn ADR."
    )


def render_medical_output(output: MedicalRAGOutput):
    st.markdown("### Medical RAG")

    st.metric("RAG status", output.status)
    st.caption(f"Generation model: {output.generation_model}")

    if output.status == "answered":
        st.markdown("#### Câu trả lời dựa trên evidence")

        for claim in output.claims:
            st.markdown(claim.text)
            st.caption(
                "Evidence IDs: " + ", ".join(claim.evidence_ids)
            )
    else:
        st.warning(output.answer_text)

    st.markdown("#### Retrieved evidence")

    if not output.retrieved_evidence:
        st.info("Không có evidence được truy xuất.")
    else:
        for evidence in output.retrieved_evidence:
            data = evidence.model_dump(mode="json")

            evidence_id = data.get("evidence_id", "Evidence")
            title = (
                data.get("title")
                or data.get("source_name")
                or evidence_id
            )

            with st.expander(f"{evidence_id} — {title}"):
                content = (
                    data.get("content")
                    or data.get("excerpt")
                    or data.get("text")
                )

                if content:
                    st.write(content)

                source_url = data.get("source_url")
                if source_url:
                    st.markdown(f"[Mở nguồn tài liệu]({source_url})")

                if data.get("score") is not None:
                    st.caption(f"Similarity score: {data['score']:.4f}")

                st.json(data)

    if output.citations:
        st.markdown("#### Citations used by generated claims")
        for citation in output.citations:
            data = citation.model_dump(mode="json")
            st.write(
                f"{data.get('evidence_id', 'Evidence')} — "
                f"{data.get('source_name', data.get('title', 'Source'))}"
            )


def render_execution_result(result: AgentExecutionResult):
    st.write(f"Agent: `{result.agent}`")
    st.write(f"Execution status: `{result.status}`")

    if result.status == "failed":
        st.error(result.error)

    elif result.status == "not_implemented":
        st.error(result.error)

    elif result.status == "skipped":
        st.warning(result.skipped_reason)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Healthcare — Multi-Agent Demo",
    page_icon="🩺",
    layout="wide",
)

st.title("AI Healthcare — Multi-Agent Demo v0.1")

st.caption(
    "Commander → Symptom Agent → Medical RAG → Response Assembly"
)

st.warning(
    "Research prototype only. Không sử dụng thông tin sức khỏe thật. "
    "Kết quả không phải chẩn đoán hoặc hướng dẫn điều trị."
)

with st.sidebar:
    st.markdown("### Runtime configuration")
    st.write(f"LLM: `{OLLAMA_MODEL}`")
    st.write(f"Embedding: `{EMBEDDING_MODEL}`")
    st.write(f"Qdrant: `{QDRANT_URL}`")
    st.write(f"Collection: `{COLLECTION_NAME}`")

    st.caption(
        "Demo chỉ bật Symptom Agent và Medical RAG Agent."
    )

with st.form("symptom_demo_form"):
    question = st.text_area(
        "Mô tả triệu chứng",
        placeholder=(
            "Ví dụ: Tôi bị đau đầu và buồn nôn từ sáng nay..."
        ),
        height=140,
    )

    submitted = st.form_submit_button(
        "Phân tích",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not question.strip():
        st.info("Vui lòng nhập nội dung trước khi gửi.")

    else:
        start_time = time.perf_counter()

        try:
            with st.spinner(
                "Đang khởi tạo runtime và phân tích yêu cầu..."
            ):
                runtime = get_runtime()

                commander_plan = runtime["commander"].plan(
                    question.strip()
                )

            st.markdown("## 1. Commander routing")
            st.write(f"Intent: `{commander_plan.intent}`")
            st.write(f"Plan status: `{commander_plan.status}`")

            with st.expander("Xem Commander plan"):
                st.json(commander_plan.model_dump(mode="json"))

            if commander_plan.status == "needs_clarification":
                st.info(commander_plan.clarification_question)

            elif commander_plan.status == "unsupported":
                st.warning(
                    "Yêu cầu này chưa được hỗ trợ trong demo v0.1."
                )

            elif (
                commander_plan.intent != "symptom_question"
                or len(commander_plan.tasks) != 1
                or commander_plan.tasks[0].agent != "symptom"
            ):
                st.warning(
                    "Demo v0.1 hiện chỉ tiếp nhận yêu cầu "
                    "được Commander định tuyến riêng cho Symptom Agent."
                )

            else:
                symptom_task = commander_plan.tasks[0]

                next_task_number = max(
                    int(task.task_id.split("_")[1])
                    for task in commander_plan.tasks
                ) + 1

                # RAG is a mandatory system-level grounding step
                # after successful symptom extraction.
                rag_task = AgentTask(
                    task_id=f"task_{next_task_number}",
                    agent="medical_rag",
                    objective=(
                        "Retrieve and generate medical information "
                        "grounded in evidence using the extracted symptoms."
                    ),
                    depends_on=(symptom_task.task_id,),
                )

                # Internal workflow includes two agents.
                # The original Commander intent remains symptom_question.
                workflow_plan = CommanderPlan(
                    intent="mixed",
                    status="ready",
                    tasks=(symptom_task, rag_task),
                    clarification_question=None,
                )

                with st.spinner(
                    "Commander đã định tuyến. "
                    "Đang chạy Symptom Agent và Medical RAG..."
                ):
                    results = runtime["orchestrator"].execute_plan(
                        workflow_plan,
                        context={
                            "request_id": uuid.uuid4().hex,
                            "user_message": question.strip(),
                        },
                    )

                    assembled = runtime["assembler"].assemble(results)

                st.markdown("## 2. Agent execution")
                st.write(f"Workflow status: `{assembled.status}`")

                for result in results:
                    with st.expander(
                        f"{result.agent} — {result.status}",
                        expanded=True,
                    ):
                        render_execution_result(result)

                        if (
                            result.status == "succeeded"
                            and result.agent == "symptom"
                            and isinstance(
                                result.output,
                                SymptomExtractionOutput,
                            )
                        ):
                            render_symptom_output(result.output)

                        elif (
                            result.status == "succeeded"
                            and result.agent == "medical_rag"
                            and isinstance(
                                result.output,
                                MedicalRAGOutput,
                            )
                        ):
                            render_medical_output(result.output)

                st.markdown("## 3. Runtime metrics")

                elapsed = time.perf_counter() - start_time

                col1, col2, col3 = st.columns(3)
                col1.metric("Total latency", f"{elapsed:.2f}s")
                col2.metric("Successful agents", assembled.succeeded_count)
                col3.metric("Failed agents", assembled.failed_count)

                st.caption(
                    f"Qdrant collection: {runtime['collection']} · "
                    f"Indexed points: {runtime['point_count']}"
                )

        except CommanderPolicyError:
            st.warning(
                "Commander chọn chức năng ngoài phạm vi demo hiện tại. "
                "Phiên bản này chỉ bật Symptom và Medical RAG."
            )

        except Exception as exc:
            st.error(
                f"Không thể hoàn thành workflow: "
                f"{type(exc).__name__}: {exc}"
            )

st.divider()

st.caption(
    "AI Healthcare research prototype. "
    "Retrieval metrics, model outputs and clinical correctness "
    "are separate evaluation targets."
)
