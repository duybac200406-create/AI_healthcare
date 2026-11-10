"""End-to-end smoke test using real BGE-M3 through Ollama and Qdrant Server.

This indexes only the synthetic latex-allergy evidence fixture. It does not
call an LLM and does not produce medical advice.
"""

from pathlib import Path

from ai_healthcare.ingestion.indexer import DocumentIndexer
from ai_healthcare.ingestion.synthetic_evidence_loader import load_synthetic_evidence
from ai_healthcare.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from ai_healthcare.retrieval.dense_retriever import DenseRetriever
from ai_healthcare.schemas.vector import VectorSearchFilter, VectorStoreConfig
from ai_healthcare.vectorstores.qdrant_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = PROJECT_ROOT / "data" / "synthetic" / "evidence" / "latex_allergy.json"


def main() -> None:
    if not EVIDENCE_PATH.is_file():
        raise FileNotFoundError(
            "Synthetic evidence fixture was not found at the expected project path."
        )

    documents = load_synthetic_evidence(EVIDENCE_PATH)
    config = VectorStoreConfig(
        collection_name="medical_knowledge_bge_m3_smoke",
        vector_size=1024,
        distance="cosine",
        embedding_model="bge-m3",
    )

    with OllamaEmbeddingProvider(model="bge-m3") as embedding_provider:
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
                batch_size=16,
            )

            print(f"Loaded synthetic evidence documents: {len(documents)}")
            for document in documents:
                ingestion = indexer.index_document(document)
                print(
                    "INDEXED:",
                    document.document_id,
                    "chunks=", ingestion.chunk_count,
                    "fingerprint=", ingestion.pipeline_fingerprint[:12],
                )

            retriever = DenseRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            queries = (
                "Phương pháp nào được sử dụng để chẩn đoán dị ứng latex?",
                "Người bị dị ứng latex nên làm gì để tránh tiếp xúc?",
            )
            filters = VectorSearchFilter(languages=("vi",))

            for query in queries:
                print(f"\nQUERY: {query}")
                results = retriever.retrieve(
                    query,
                    top_k=3,
                    filters=filters,
                )
                if not results:
                    print("NO RESULTS")
                    continue

                for rank, result in enumerate(results, start=1):
                    print(
                        f"{rank}. score={result.score:.4f} "
                        f"document_id={result.document_id} "
                        f"chunk_id={result.chunk_id} "
                        f"source_type={result.source_type}"
                    )
                    print(f"   {result.content}")

            print("\nDENSE RETRIEVAL SMOKE TEST COMPLETED")
            print(
                "NOTE: Synthetic fixture retrieval validates plumbing only; "
                "it does not establish medical retrieval quality."
            )


if __name__ == "__main__":
    main()
