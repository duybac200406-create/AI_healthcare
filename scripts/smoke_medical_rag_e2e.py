"""End-to-end smoke test with a real public-domain MedlinePlus source.

This exercises real BGE-M3 embeddings, Qdrant Server, Qwen3 structured
outputs, MedicalRAGAgent, Orchestrator, and ResponseAssembler. It is a
plumbing/integration check, not a clinical-quality evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_healthcare.agents.medical_rag import MedicalRAGAgent
from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.orchestration.response_assembler import ResponseAssembler
from ai_healthcare.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from ai_healthcare.providers.ollama_provider import OllamaProvider
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.agent import AgentTask, CommanderPlan
from ai_healthcare.schemas.document import MedicalDocument
from ai_healthcare.schemas.medical_response import MedicalRAGOutput
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "samples" / "medlineplus_latex_allergy.json"
COLLECTION_NAME = "medical_knowledge_medlineplus_e2e"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen3:8b"
QUESTION = "Những phương pháp nào có thể được bác sĩ sử dụng để chẩn đoán dị ứng latex?"


def load_source_fixture() -> tuple[MedicalDocument, dict[str, Any]]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Source fixture not found: {SOURCE_PATH}")

    try:
        payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read the MedlinePlus source fixture.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Source fixture must be a JSON object.")

    source_metadata = payload.get("source_metadata")
    document_payload = payload.get("document")
    if not isinstance(source_metadata, dict) or not isinstance(document_payload, dict):
        raise ValueError("Source fixture must contain source_metadata and document objects.")

    if source_metadata.get("copyright_status") != "public_domain":
        raise ValueError("Source fixture is not marked as public-domain content.")
    if source_metadata.get("attribution") != "Source: MedlinePlus, National Library of Medicine.":
        raise ValueError("Required MedlinePlus attribution is missing.")

    # retrieved_at records when this local ingestion run reads the source;
    # it is not the page publication/update date.
    document_payload = dict(document_payload)
    document_payload.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    document = MedicalDocument.model_validate(document_payload)
    if str(document.source_url) != "https://medlineplus.gov/latexallergy.html":
        raise ValueError("Unexpected source URL in the MedlinePlus fixture.")
    if document.source_type != "other":
        raise ValueError("The MedlinePlus Health Topic summary must not be mislabeled as a clinical guideline.")
    if document.publication_date is not None:
        raise ValueError("The page's last-updated date must not be mislabeled as publication_date.")

    return document, source_metadata


def main() -> None:
    document, source_metadata = load_source_fixture()
    config = VectorStoreConfig(
        collection_name=COLLECTION_NAME,
        vector_size=1024,
        distance="cosine",
        embedding_model=EMBEDDING_MODEL,
    )

    print("SOURCE:", document.title)
    print("SOURCE NAME:", document.source_name)
    print("SOURCE URL:", document.source_url)
    print("SOURCE LAST UPDATED:", source_metadata["page_last_updated"])
    print("SOURCE TYPE:", document.source_type)
    print("ATTRIBUTION:", source_metadata["attribution"])

    with OllamaEmbeddingProvider(model=EMBEDDING_MODEL) as embedding_provider:
        with QdrantVectorStore(
            url="http://127.0.0.1:6333",
            config=config,
        ) as vector_store:
            vector_store.ensure_collection()

            indexer = DocumentIndexer(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                max_chars=1200,
                overlap_chars=150,
                batch_size=8,
            )
            ingestion = indexer.index_document(document)
            print("INDEXED DOCUMENT:", ingestion.document_id)
            print("CHUNK COUNT:", ingestion.chunk_count)
            print("PIPELINE FINGERPRINT:", ingestion.pipeline_fingerprint)

            retriever = DenseRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            retrieval_filter = VectorSearchFilter(
                document_ids=(document.document_id,),
            )
            retrieval_results = retriever.retrieve(
                QUESTION,
                top_k=3,
                filters=retrieval_filter,
            )
            if not retrieval_results:
                raise RuntimeError("Dense retrieval returned no chunks for the real source.")

            print("\nRETRIEVAL RESULTS:")
            for rank, result in enumerate(retrieval_results, start=1):
                print(
                    f"{rank}. score={result.score:.4f} "
                    f"document_id={result.document_id} "
                    f"chunk_id={result.chunk_id} "
                    f"source_type={result.source_type}"
                )
                print("   SOURCE:", result.source_name)
                print("   URL:", result.source_url)
                print("   EXCERPT:", result.content[:500])

            with OllamaProvider(model=LLM_MODEL) as llm_provider:
                medical_agent = MedicalRAGAgent(
                    provider=llm_provider,
                    retriever=retriever,
                    top_k=3,
                )
                registry = AgentRegistry()
                registry.register(medical_agent)
                orchestrator = Orchestrator(
                    enabled_agents={"medical_rag"},
                    registry=registry,
                )

                task = AgentTask(
                    task_id="task_1",
                    agent="medical_rag",
                    objective="Answer the general question using the retrieved MedlinePlus evidence.",
                )
                plan = CommanderPlan(
                    intent="medical_information",
                    status="ready",
                    tasks=(task,),
                )

                execution_results = orchestrator.execute_plan(
                    plan,
                    context={"user_message": QUESTION},
                )
                assembled = ResponseAssembler().assemble(execution_results)

                if assembled.status != "completed":
                    raise RuntimeError(
                        f"Orchestrator execution did not complete: {assembled.status}"
                    )

                output = execution_results[0].output
                if not isinstance(output, MedicalRAGOutput):
                    raise RuntimeError("MedicalRAGAgent returned an unexpected output type.")

                print("\nMEDICAL RAG STATUS:", output.status)
                print("ANSWER:")
                print(output.answer_text)

                if output.status == "answered":
                    expected_url = str(document.source_url)
                    if not output.citations:
                        raise RuntimeError("Answered output contains no citations.")
                    for citation in output.citations:
                        if citation.source_url != expected_url:
                            raise RuntimeError("Citation URL does not match the indexed source.")
                        if citation.source_type == "synthetic_fixture":
                            raise RuntimeError("Synthetic evidence must not support this answer.")
                        print(
                            "CITATION:",
                            citation.evidence_id,
                            citation.source_name,
                            citation.title,
                            citation.source_url,
                        )
                    print("\nEND-TO-END MEDICAL RAG: PASS")
                elif output.status in {
                    "no_evidence",
                    "insufficient_evidence",
                    "conflicting_evidence",
                }:
                    print("\nSAFE ABSTENTION: output contract was respected.")
                    print("NOTE: inspect retrieval and model response; the question is directly covered by this source.")
                else:
                    raise RuntimeError(
                        f"MedicalRAGAgent rejected generation: {output.status}"
                    )


if __name__ == "__main__":
    main()
