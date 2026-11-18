"""Shared runtime construction for the Streamlit app and the smoke script.

Builds Commander, Symptom Agent, Medical RAG and the conversation service
against an EXISTING Qdrant collection. It never creates or modifies a
collection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from ai_healthcare.agents.commander import Commander
from ai_healthcare.agents.medical_rag import MedicalRAGAgent
from ai_healthcare.agents.symptom import SymptomAgent
from ai_healthcare.orchestration.agent_registry import AgentRegistry
from ai_healthcare.orchestration.orchestrator import Orchestrator
from ai_healthcare.providers.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)
from ai_healthcare.providers.ollama_provider import OllamaProvider
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.vector import VectorStoreConfig
from ai_healthcare.services.conversation_service import ConversationService
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLLECTION = "medical_knowledge_bge_m3_symptom_4b_lite_plaintext_v1"
DEFAULT_RAG_PROMPT = REPO_ROOT / "prompts" / "medical_rag_system_v2.md"


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "bge-m3"
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = DEFAULT_COLLECTION
    rag_prompt_path: Path = DEFAULT_RAG_PROMPT
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        env = os.environ.get
        return cls(
            ollama_url=env("AI_HEALTHCARE_OLLAMA_URL", cls.ollama_url),
            chat_model=env("AI_HEALTHCARE_CHAT_MODEL", cls.chat_model),
            qdrant_url=env("AI_HEALTHCARE_QDRANT_URL", cls.qdrant_url),
            collection=env("AI_HEALTHCARE_QDRANT_COLLECTION", DEFAULT_COLLECTION),
            rag_prompt_path=Path(
                env("AI_HEALTHCARE_RAG_PROMPT_PATH", str(DEFAULT_RAG_PROMPT))
            ),
        )


@dataclass
class Runtime:
    service: ConversationService
    config: RuntimeConfig
    point_count: int
    closers: tuple = ()

    def close(self) -> None:
        for closer in self.closers:
            try:
                closer()
            except Exception:
                pass


def build_runtime(config: RuntimeConfig | None = None) -> Runtime:
    config = config or RuntimeConfig.from_env()

    # Check before ensure_collection(), which could create a missing one.
    probe = QdrantClient(url=config.qdrant_url, timeout=10.0)
    try:
        if not probe.collection_exists(config.collection):
            raise RuntimeError(
                f"Qdrant collection '{config.collection}' does not exist. "
                "Refusing to create an empty collection automatically."
            )
        point_count = probe.count(
            collection_name=config.collection, exact=True
        ).count
        if point_count <= 0:
            raise RuntimeError(
                f"Qdrant collection '{config.collection}' is empty."
            )
    finally:
        probe.close()

    rag_prompt = config.rag_prompt_path.read_text(encoding="utf-8")

    llm = OllamaProvider(base_url=config.ollama_url, model=config.chat_model)
    embedder = OllamaEmbeddingProvider(
        base_url=config.ollama_url, model=config.embedding_model
    )
    store = QdrantVectorStore(
        url=config.qdrant_url,
        config=VectorStoreConfig(
            collection_name=config.collection,
            vector_size=1024,
            distance="cosine",
            embedding_model=config.embedding_model,
        ),
    )

    try:
        store.ensure_collection()  # validates the existing configuration
        retriever = DenseRetriever(embedding_provider=embedder, vector_store=store)

        symptom_agent = SymptomAgent(provider=llm)
        rag_agent = MedicalRAGAgent(
            provider=llm,
            retriever=retriever,
            top_k=config.top_k,
            system_prompt=rag_prompt,
        )

        registry = AgentRegistry()
        registry.register(symptom_agent)
        registry.register(rag_agent)
        orchestrator = Orchestrator(
            enabled_agents={"symptom", "medical_rag"}, registry=registry
        )
        commander = Commander(provider=llm, orchestrator=orchestrator)

        service = ConversationService(
            commander=commander,
            symptom_agent=symptom_agent,
            medical_rag_agent=rag_agent,
        )
    except Exception:
        store.close()
        embedder.close()
        llm.close()
        raise

    return Runtime(
        service=service,
        config=config,
        point_count=point_count,
        closers=(store.close, embedder.close, llm.close),
    )
