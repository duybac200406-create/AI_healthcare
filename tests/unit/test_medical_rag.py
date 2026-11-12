import json
from ai_healthcare.providers.base import GenerationConfig
import pytest
from pydantic import ValidationError

from ai_healthcare.agents.medical_rag import (
    MedicalRAGAgent,
    MedicalRAGGenerationError,
    MedicalRAGInputError,
)
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.schemas.agent import AgentTask, CommanderPlan
from ai_healthcare.schemas.medical_response import MedicalGeneration, MedicalRAGOutput
from ai_healthcare.schemas.vector import VectorSearchResult


class FakeProvider:
    model = "qwen3:8b"

    def __init__(self, output):
        self.output = output
        self.calls = []
        self.configs = []

    def generate_json(self, system_prompt, user_prompt, json_schema,*,config=None):
        self.configs.append(config)
        self.calls.append((system_prompt, user_prompt, json_schema))
        return self.output


class FakeRetriever:
    def __init__(self, results=()):
        self.results = tuple(results)
        self.calls = []

    def retrieve(self, query, *, top_k=5, filters=None, score_threshold=None):
        self.calls.append((query, top_k, filters, score_threshold))
        return self.results


def make_result(
    *,
    document_id="doc_guideline_1",
    chunk_id="chunk_guideline_1_0000",
    content="Khám lâm sàng và xét nghiệm có thể được sử dụng để đánh giá dị ứng latex.",
    score=0.81,
    source_type="clinical_guideline",
):
    return VectorSearchResult(
        point_id="7c5c9b1e-81af-5eb6-8e94-ecf1a1a2ed81",
        score=score,
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_index=0,
        content=content,
        title="Latex allergy assessment",
        source_type=source_type,
        source_name="Synthetic test source" if source_type == "synthetic_fixture" else "Clinical source",
        source_url="https://example.org/guideline" if source_type != "synthetic_fixture" else None,
        doi=None,
        publication_date="2025-01-01",
        retrieved_at="2026-11-08T00:00:00+00:00",
        language="vi",
        license=None,
        section_title="Assessment",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=len(content),
        raw_content_sha256="a" * 64,
        normalized_content_sha256="b" * 64,
        normalizer_version="normalizer-v1",
        chunker_version="char-chunker-v1",
        pipeline_fingerprint="c" * 64,
        embedding_model="bge-m3",
        embedding_dimensions=1024,
    )


def make_task(agent="medical_rag"):
    return AgentTask(
        task_id="task_1",
        agent=agent,
        objective="Trả lời câu hỏi thông tin y khoa dựa trên bằng chứng.",
    )


def make_generation(*, status="answered", claims=None, reason=None):
    return json.dumps(
        {
            "status": status,
            "claims": claims if claims is not None else (
                [
                    {
                        "text": "Tài liệu nêu rằng khám lâm sàng và xét nghiệm có thể được sử dụng.",
                        "evidence_ids": ["E1"],
                    }
                ] if status == "answered" else []
            ),
            "reason": reason if status != "answered" else None,
        },
        ensure_ascii=False,
    )


def make_agent(provider_output=None, results=(), **kwargs):
    provider = FakeProvider(provider_output or make_generation())
    retriever = FakeRetriever(results)
    agent = MedicalRAGAgent(
        provider=provider,
        retriever=retriever,
        **kwargs,
    )
    return agent, provider, retriever


def test_agent_name_is_registered_name():
    agent, _, _ = make_agent()
    assert agent.name == "medical_rag"


def test_answered_output_uses_only_validated_retrieved_citations():
    result = make_result()
    agent, provider, retriever = make_agent(results=[result])

    output = agent.execute(make_task(), {"user_message": "Dị ứng latex được đánh giá thế nào?"})

    assert isinstance(output, MedicalRAGOutput)
    assert output.status == "answered"
    assert output.answer_text.startswith("Tài liệu nêu rằng")
    assert [item.evidence_id for item in output.citations] == ["E1"]
    assert output.citations[0].source_url == "https://example.org/guideline"
    assert output.citations[0].document_id == result.document_id
    assert len(provider.calls) == 1
    assert retriever.calls[0][0] == "Dị ứng latex được đánh giá thế nào?"



def test_agent_registers_and_runs_through_existing_orchestrator():
    agent, _, _ = make_agent(results=[make_result()])
    registry = AgentRegistry()
    registry.register(agent)
    orchestrator = Orchestrator(registry=registry)
    plan = CommanderPlan(
        intent="medical_information",
        status="ready",
        tasks=(make_task(),),
    )

    results = orchestrator.execute_plan(
        plan,
        context={"user_message": "Dị ứng latex được đánh giá thế nào?"},
    )

    assert len(results) == 1
    assert results[0].status == "succeeded"
    assert isinstance(results[0].output, MedicalRAGOutput)
    assert results[0].output.status == "answered"

def test_no_retrieved_evidence_abstains_without_calling_llm():
    agent, provider, retriever = make_agent(results=[])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "no_evidence"
    assert output.abstention_reason == "no_evidence"
    assert output.answer_text == "Tôi chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi này."
    assert provider.calls == []
    assert len(retriever.calls) == 1


def test_unknown_evidence_id_fails_closed():
    bad_output = make_generation(
        claims=[{"text": "Một khẳng định không hợp lệ.", "evidence_ids": ["E99"]}]
    )
    agent, _, _ = make_agent(provider_output=bad_output, results=[make_result()])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "generation_rejected"
    assert output.abstention_reason == "invalid_citations"
    assert output.claims == ()
    assert output.citations == ()
    assert len(output.retrieved_evidence) == 1


def test_multiple_claims_produce_citations_in_retrieval_order():
    second = make_result(
        document_id="doc_guideline_2",
        chunk_id="chunk_guideline_2_0000",
        content="Nên hạn chế tiếp xúc với nguồn latex.",
        score=0.72,
    )
    generated = make_generation(
        claims=[
            {"text": "Claim one.", "evidence_ids": ["E1"]},
            {"text": "Claim two.", "evidence_ids": ["E2"]},
        ]
    )
    agent, _, _ = make_agent(
        provider_output=generated,
        results=[make_result(), second],
    )

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "answered"
    assert [item.evidence_id for item in output.citations] == ["E1", "E2"]
    assert len(output.claims) == 2


def test_duplicate_retrieval_results_are_deduplicated_before_generation():
    result = make_result()
    agent, provider, _ = make_agent(results=[result, result])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert len(output.retrieved_evidence) == 1
    payload = json.loads(provider.calls[0][1])
    assert [item["evidence_id"] for item in payload["retrieved_evidence"]] == ["E1"]


def test_prompt_payload_does_not_expose_similarity_score_or_internal_hashes():
    agent, provider, _ = make_agent(results=[make_result()])

    agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    payload = json.loads(provider.calls[0][1])
    evidence = payload["retrieved_evidence"][0]
    assert evidence["evidence_id"] == "E1"
    assert "similarity_score" not in evidence
    assert "raw_content_sha256" not in evidence
    assert "normalized_content_sha256" not in evidence


def test_abstains_when_model_reports_insufficient_evidence():
    generated = make_generation(
        status="insufficient_evidence",
        reason="Tài liệu không nêu đủ thông tin để kết luận.",
    )
    agent, _, _ = make_agent(provider_output=generated, results=[make_result()])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "insufficient_evidence"
    assert output.claims == ()
    assert output.citations == ()
    assert len(output.retrieved_evidence) == 1


def test_abstains_when_model_reports_conflicting_evidence():
    generated = make_generation(
        status="conflicting_evidence",
        reason="Các tài liệu cung cấp thông tin chưa thống nhất.",
    )
    agent, _, _ = make_agent(provider_output=generated, results=[make_result()])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "conflicting_evidence"
    assert output.abstention_reason == "conflicting_evidence"
    assert output.claims == ()



def test_synthetic_fixture_only_evidence_never_reaches_llm():
    synthetic = make_result(source_type="synthetic_fixture")
    agent, provider, retriever = make_agent(results=[synthetic])

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    assert output.status == "insufficient_evidence"
    assert output.abstention_reason == "insufficient_evidence"
    assert len(output.retrieved_evidence) == 1
    assert output.citations == ()
    assert provider.calls == []
    assert len(retriever.calls) == 1


def test_synthetic_fixture_is_excluded_when_real_evidence_is_also_present():
    synthetic = make_result(
        document_id="doc_synthetic_1",
        chunk_id="chunk_synthetic_1_0000",
        source_type="synthetic_fixture",
    )
    real = make_result(
        document_id="doc_guideline_2",
        chunk_id="chunk_guideline_2_0000",
        content="Nguồn lâm sàng mô tả cách đánh giá dị ứng latex.",
        score=0.70,
    )
    generated = make_generation(
        claims=[{"text": "Claim dựa trên nguồn thật.", "evidence_ids": ["E2"]}]
    )
    agent, provider, _ = make_agent(
        provider_output=generated,
        results=[synthetic, real],
    )

    output = agent.execute(make_task(), {"user_message": "Câu hỏi y khoa"})

    prompt_payload = json.loads(provider.calls[0][1])
    assert [item["evidence_id"] for item in prompt_payload["retrieved_evidence"]] == ["E2"]
    assert [item.evidence_id for item in output.citations] == ["E2"]
    assert [item.source_type for item in output.retrieved_evidence] == [
        "synthetic_fixture", "clinical_guideline"
    ]

def test_rejects_patient_record_context():
    agent, provider, retriever = make_agent(results=[make_result()])

    with pytest.raises(MedicalRAGInputError, match="does not accept patient-record"):
        agent.execute(
            make_task(),
            {
                "user_message": "Câu hỏi y khoa",
                "patient_record": {"allergy": "latex"},
            },
        )

    assert provider.calls == []
    assert retriever.calls == []


@pytest.mark.parametrize("key", ["patient_data", "patient_profile", "patient_context", "ehr", "emr_record"])
def test_rejects_patient_related_context_keys(key):
    agent, _, _ = make_agent(results=[make_result()])

    with pytest.raises(MedicalRAGInputError):
        agent.execute(make_task(), {"user_message": "Question", key: {"x": 1}})


def test_requires_user_message_in_context():
    agent, _, retriever = make_agent(results=[make_result()])

    with pytest.raises(MedicalRAGInputError, match="user_message"):
        agent.execute(make_task(), {})

    assert retriever.calls == []


def test_rejects_task_for_another_agent():
    agent, _, _ = make_agent()

    with pytest.raises(MedicalRAGInputError, match="another agent"):
        agent.execute(make_task(agent="symptom"), {"user_message": "Question"})


def test_rejects_blank_and_oversized_queries():
    agent, _, retriever = make_agent(results=[make_result()], max_query_chars=5)

    with pytest.raises(MedicalRAGInputError, match="blank"):
        agent.execute(make_task(), {"user_message": "  "})
    with pytest.raises(MedicalRAGInputError, match="length limit"):
        agent.execute(make_task(), {"user_message": "123456"})

    assert retriever.calls == []


def test_malformed_or_schema_invalid_model_output_raises_safe_error():
    invalid_outputs = [
        "not-json",
        json.dumps({"status": "answered", "claims": [], "reason": None}),
    ]
    for invalid in invalid_outputs:
        agent, _, _ = make_agent(provider_output=invalid, results=[make_result()])
        with pytest.raises(MedicalRAGGenerationError, match="MedicalGeneration contract"):
            agent.execute(make_task(), {"user_message": "Question"})


def test_system_prompt_contains_json_schema_and_source_trust_boundaries():
    agent, provider, _ = make_agent(results=[make_result()])

    agent.execute(make_task(), {"user_message": "Question"})

    system_prompt = provider.calls[0][0]
    assert "Required MedicalGeneration JSON Schema" in system_prompt
    assert "untrusted reference data" in system_prompt
    assert "Synthetic fixtures are test data" in system_prompt


def test_generated_claim_requires_at_least_one_evidence_id():
    with pytest.raises(ValidationError):
        MedicalGeneration.model_validate(
            {"status": "answered", "claims": [{"text": "Claim", "evidence_ids": []}], "reason": None}
        )
def test_medical_rag_passes_generation_config_to_provider():
    config = GenerationConfig(
        temperature=0.2,
        num_ctx=8192,
        num_predict=512,
    )

    agent, provider, _ = make_agent(
        results=[make_result()],
        generation_config=config,
    )

    agent.execute(
        make_task(),
        {"user_message": "Dị ứng latex được đánh giá thế nào?"},
    )

    assert provider.configs == [config]