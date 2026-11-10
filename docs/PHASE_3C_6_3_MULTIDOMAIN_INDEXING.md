# Phase 3C.6.3 – Multi-domain Indexing

## Purpose

Index the seven captured MedlinePlus Health Topic Summary documents into a dedicated Qdrant collection. This is a technical and retrieval pilot, not a comprehensive or clinically validated medical knowledge base.

## Prerequisites

- Seven JSON files in `data/samples/medlineplus_health_topics/`.
- Qdrant Server running at `http://127.0.0.1:6333`.
- Ollama running with `bge-m3` available.
- `qdrant-client==1.20.0` installed in the project `.venv`.

## Collection

- Name: `medical_knowledge_bge_m3_multidomain_pilot_v1`
- Dense vector size: 1024
- Distance: Cosine
- Embedding model identifier: `bge-m3`
- Patient records: excluded

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts/index_medlineplus_multidomain.py
```

The script validates all seven JSON files before opening the Qdrant write path. It then uses the existing `DocumentIndexer`, BGE-M3 embedding provider and Qdrant adapter. It prints active chunk counts per domain.

## Repeatability and recovery

The script is idempotent per document: rerunning uses the existing `DocumentIndexer` and deterministic pipeline/point identities. If indexing stops partway through the seven documents, previously completed documents remain indexed; rerun after resolving the failure. Do not assume the seven-document corpus was atomically committed as one transaction.

## Scope limits

- Only captured MedlinePlus `FullSummary` content is indexed.
- The source metadata is not converted into clinical recommendations.
- The seven-domain sample is not evidence of broad medical coverage.
- Similarity score is not a probability, diagnosis, or evidence-sufficiency score.
