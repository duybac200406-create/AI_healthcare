# AI Healthcare — Project Process & Progress

> **Last updated:** 2026-11-10  
> **Purpose:** Track implementation stages, verified outcomes, decisions, and next work.  
> **Truth mode:** A task is marked complete only when its output or test evidence exists. A design proposal is not an implementation, and a retrieval metric is not proof of clinical correctness.

## 1. Overall status

**Current checkpoint: Stage 4 completed; Stage 5 is next.**

The data/index/retrieval foundation has been built and audited. The end-to-end medical assistant (generation, safety, multi-agent orchestration, and user-facing workflow) is **not complete**.

### Stage overview

| Stage | Scope | Status |
|---|---|---|
| 1 | Problem definition, research scope, and initial architecture | Completed (planning baseline) |
| 2 | Corpus and initial benchmark construction | Completed for the current pilot |
| 3 | Ingestion, normalization/chunking, embeddings, and Qdrant indexing | Completed for two pilot collections |
| 4 | Provenance checks, live Qdrant payload audit, and document-level retrieval evaluation | Completed for the current pilot |
| 5 | Evidence-level retrieval evaluation | **Next / in progress not yet started** |
| 6 | Answer generation, grounding, citation, and safety/abstention evaluation | Pending |
| 7 | Multi-agent implementation and integration | Pending; architecture is defined, full end-to-end operation not verified |
| 8 | End-to-end validation, reproducibility, documentation, and demo | Pending |

This is a project milestone map, not a claim that the project is 50% or any other exact percentage complete.

## 2. Verified pilot assets

### Corpus
- Source: MedlinePlus FullSummary Health Topics.
- Domains: Heart Diseases, Type 2 Diabetes, COPD, GERD, CKD, Migraine, Depression (7 domains).
- Manifest: `data/manifests/medlineplus_multidomain_pilot_v0.2.json`.
- Benchmark: `data/evaluation/medical_rag_multidomain_questions_v0.1.json`.
- Benchmark size: 28 in-scope/answerable pilot questions + 7 out-of-scope diagnostics.
- Plaintext sources: `data/samples/medlineplus_health_topics/`.
- HTML backup: `data/samples/medlineplus_health_topics_html_backup_20261108/`.

### Qdrant collections (preserve/freeze for current comparison)
1. `medical_knowledge_bge_m3_multidomain_pilot_v1`
   - 45 points: 38 chunks + 7 document-state points.
   - Source snapshot SHA-256: `c915ea7cd24d996ac7722930df0bd34a82a3824089c4fb40abe8a0082e8c1b96`.
   - Index identity: `bbc9c24a48935787240aabbbaac6acff7fb35cc4d410c19600be3f40baf53948`.
   - Source directory: HTML backup.

2. `medical_knowledge_bge_m3_multidomain_plaintext_v1`
   - 39 points: 32 chunks + 7 document-state points.
   - Source snapshot SHA-256: `7bb45e1506192a2b8f1137ced6fb6db54b6e37492d8fee7bbfd2a8ea182a7a99`.
   - Index identity: `6ad0fdc590854b4f93f39a702ddab3c598eb609e65bfd7a451a3480cd71aa4cf`.
   - Source directory: plaintext sources.

Both use BGE-M3 embeddings (1024 dimensions), Cosine, `normalizer-v1`, `char-chunker-v1`, max chunk size 1200 characters, overlap 150. Their provenance sidecars are:
- `data/manifests/index_builds/baseline_html_v1.provenance.json`
- `data/manifests/index_builds/plaintext_v1.provenance.json`

**Important provenance caveat:** sidecars are retrospectively verified. Historical embedding-model digest and indexing-time Git commit are unknown; current model digest must not be presented as the historical digest.

### Verified retrieval results (28 in-scope questions)
| Metric | HTML baseline | Plaintext |
|---|---:|---:|
| Hit@1 | 26/28 (92.9%) | 26/28 (92.9%) |
| Hit@3 | 27/28 (96.4%) | 28/28 (100%) |
| Hit@5 | 28/28 (100%) | 28/28 (100%) |
| MRR@5 | 0.9536 | 0.9643 |
| Top-1 domain accuracy | 92.9% | 92.9% |
| Known-domain-filtered MRR | 1.0000 | 1.0000 |

Reports:
- `data/evaluation/results/multidomain_retrieval_baseline_live_audit_20261108.json`
- `data/evaluation/results/multidomain_retrieval_plaintext_live_audit_20261108.json`

Interpretation limits:
- Hit@5 only says the expected document is in the top 5; it does not establish answer correctness.
- Known-domain filtering uses the benchmark's gold domain (oracle information); it is not evidence of autonomous domain-classification accuracy.
- 28 questions (4/domain) are a small pilot, not a clinically representative benchmark.
- The metric name `domain_filter_leakage_in_top_5` may be misleading in unfiltered mode; it counts off-target-domain results in top-K, not leakage from an active filter.

### Evidence annotation pilot
- File: `data/evaluation/medical_rag_evidence_annotations_renal_pilot_v0.1.json`.
- Two CKD evidence spans have exact source offsets and quote checks.
- HTML baseline: evidence Hit@1 0/2, Hit@3 1/2, Hit@5 2/2, MRR@5 0.35.
- Plaintext: evidence Hit@1 0/2, Hit@3 2/2, Hit@5 2/2, MRR@5 0.50.
- This two-case result is a pilot only; it is not enough to generalize across all seven domains.

## 3. Model and agent decisions

### Verified
- **Commander:** Qwen3-8B through Ollama; previously run and routing-tested.
- **Embedding:** BGE-M3 through Ollama; 1024-dimensional vectors used by the current indexes.
- **Vector database:** Qdrant.
- **LLM runtime:** Ollama.

### Four child agents (agreed functional responsibilities)
1. **Medical RAG Agent:** retrieve medical knowledge and produce evidence-grounded responses.
2. **Symptom Agent:** handle symptom information, clarification, and structured symptom reasoning.
3. **Medication Agent:** handle medication questions within permitted scope and safety constraints.
4. **Psychology Support Agent:** provide supportive mental-health conversation and coping support, with escalation/safety handling where appropriate.

**Model allocation status:** The functional roles are established. The available project history does **not** verify a previously approved separate LLM model for each child agent. A practical proposal is to reuse Qwen3-8B for all four agents initially, with distinct prompts, tools, and policies; this is a proposal, not a finalized historical decision. Do not claim four separate models are already selected or implemented.

**Hardware status:** Qwen3-8B has been run on the user's machine through Ollama. That proves the model can run in the current setup, but does not by itself prove four concurrent agent calls will fit or perform acceptably. Parallelism, context length, KV-cache, VRAM/RAM, and latency still need a measured load test.

## 4. Next execution plan

### Stage 5 — Evidence Retrieval Evaluation
- Expand exact evidence annotations from the two CKD pilot cases to a balanced set across the seven domains.
- Each annotation must include source/document ID, exact quote, start/end offsets, and expected evidence/chunk relationship.
- Validate offsets against source text; reject annotations whose quote or offsets do not match.
- Report evidence Hit@1/3/5, MRR, and per-domain breakdown.
- Compare HTML and plaintext collections without changing their frozen contents.
- Categorize misses: wrong document, right document/wrong chunk, evidence split across chunks, query mismatch, or source/annotation issue.

**Exit criteria:** A reproducible evidence-level benchmark and report exist; limitations and sample size are explicit.

### Stage 6 — Answer Generation & Safety Evaluation
- Separate generation error from retrieval error by evaluating:
  1. gold/oracle evidence supplied to the model;
  2. retrieved evidence supplied to the model.
- Assess answer correctness, completeness, evidence support, citation fidelity, unsupported claims, contradictions, and abstention.
- Test insufficient-evidence and out-of-scope cases; require safe abstention rather than fabricated certainty.
- Keep evaluation non-clinical and based on fixed evidence; do not claim clinical validation.

**Exit criteria:** Reproducible answer-generation evaluation and failure taxonomy.

### Stage 7 — Multi-Agent Integration
- Implement Commander routing and the four child-agent workflows.
- Define each agent's input/output schema, system prompt, allowed tools, boundaries, and failure behavior.
- Keep patient records/EHR data separate from medical knowledge corpus; enforce access control and data minimization.
- Test handoffs, invalid routes, missing context, timeouts, and fallback behavior.
- Measure single-agent vs. multi-agent latency and resource usage before enabling concurrency.

**Exit criteria:** The four agents work through the orchestrator on a defined set of integration tests, with logs and safety boundaries.

### Stage 8 — End-to-End Validation & Demo
- Test from user input through routing, retrieval/agent processing, evidence-grounded response, and final output.
- Include normal, ambiguous, out-of-scope, insufficient-evidence, and failure cases.
- Produce reproducible setup/run/test instructions, architecture documentation, benchmark reports, and demo scenarios.
- Verify clean setup and repeatable tests.

**Exit criteria:** The project can be run and demonstrated reproducibly; research limitations are clearly stated.

## 5. Operating rules
- Do not re-index or change embedding/model settings without a documented experiment and reason.
- Keep current baseline and plaintext collections frozen for comparison.
- Do not mix retrieval quality with answer-generation quality.
- Do not equate a passing unit test, a high retrieval score, or a successful demo with clinical safety.
- Every claimed result must point to a report, test output, artifact, or reproducible command.
- Record new decisions as **Confirmed**, **Proposed**, or **Unverified**.

## 6. Immediate next checkpoint

**Start Stage 5: design and expand evidence annotations across all seven domains.**

Do not begin by changing Qdrant collections, embedding models, or the overall architecture.
