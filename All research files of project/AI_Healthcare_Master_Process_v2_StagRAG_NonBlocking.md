# AI Healthcare / Medical RAG — Master Process v2

**Version:** 2.0  
**Decision date:** 15 October 2026  
**Status:** Active — StagRAG access is no longer a project blocker  
**Purpose:** Freeze the workflow, prevent repeated literature loops, and keep work moving toward a feasible research build.

## Decision log — 15 October 2026

### Decision

Stop spending project time attempting to access the full text of **StagRAG: Explaining Where RAG Systems Go Wrong in Medical Reasoning** through the same failed routes.

Record StagRAG as:

> `FULL TEXT INACCESSIBLE IN CURRENT ATTEMPTS — DIRECT OVERLAP UNRESOLVED — NON-BLOCKING RISK`

The available abstract is enough to establish that StagRAG overlaps with generic stage-wise RAG failure diagnosis and uses oracle-guided interventions. It is **not** enough to establish whether it implements every element of the narrower controlled-evidence design. We will not claim that it does or does not.

### Consequences

- StagRAG is **not** a prerequisite for continuing Round 2, drafting a provisional research design, doing a dataset feasibility audit, or building an exploratory baseline.
- We will not claim that C2 is definitively novel relative to all prior work while this direct-overlap uncertainty remains.
- We will compare C2 against accessible, relevant full texts and report the unresolved StagRAG limitation explicitly.
- If the accessible literature demonstrates the same mechanism, we stop or pivot. If no accessible source does, we may progress provisionally and retain StagRAG as an external validity / novelty caveat rather than letting it freeze the schedule.
- We do not repeatedly retry the same IEEE/GitHub routes unless genuinely new access is available (institutional library, author manuscript, or an author response).

## Operating rules

1. The phases do not change. New papers update the decision log; they do not restart completed work.
2. No open-ended paper hunting. The current competitor set is fixed unless a paper directly challenges the research question, experimental control, or evaluation measure.
3. Inaccessible paper ≠ evidence of absence. Record uncertainty rather than guessing.
4. Separate facts, author claims, cross-paper synthesis, and our hypotheses.
5. Research novelty and project progress are different gates. An unresolved paper can prevent an absolute novelty claim without preventing feasibility tests or an exploratory prototype.
6. Do not build a patient-facing medical product for this research. The current scope is an offline evaluation/prototype for a one-semester student project.
7. No medical safety or clinical efficacy claims without appropriate clinical evaluation.

## Fixed end-to-end pipeline

| Phase | Work | Output | Current status | Exit condition |
|---|---|---|---|---|
| P0 — Scope & control | Freeze scope, contribution type, constraints, decision log | Project charter + decision log | Freeze in v2 | A single current candidate and bounded review set are named |
| P1 — Literature extraction (Parts A–D) | Existing paper extraction, evidence table, cross-paper findings, failure map | `AI Healthcare Failure Analysis.md` | **Completed** | No need to redo; only dated addenda for directly relevant papers |
| P2 — Novelty Verification Round 1 | Red-team C1–C4 and eliminate broad formulations | `Novelty Verification.md` | **Completed** | C1 STOP; C3 broad STOP; C4 crowded; C2 provisional |
| P3 — Round 2 with accessible competitors | Compare same five criteria across the bounded set; record unresolved risks | `AI_Healthcare_Round2_Competitor_Matrix_v2.md` | **In progress** | Accessible direct competitors have been checked; unresolved sources clearly marked; choose proceed / pivot / stop |
| P4 — Lock provisional RQ and hypotheses | Specify a measurable question, variables, and falsification criteria | RQ & Hypotheses sheet | **Next after P3 audit** | All terms operationalized and hypothesis can be falsified |
| P5 — Dataset feasibility | Inspect public dataset, rights, gold answers/evidence; manually audit 20–30 examples | Dataset audit + pilot annotation sheet | **Can begin in parallel with late P3** | At least one dataset supports controlled evidence conditions within budget |
| P6 — Baseline prototype | Implement no-RAG and vanilla RAG; log prompts, context, model versions, answers | Runnable evaluation harness | **May begin as an exploratory prototype** | Pipeline runs reproducibly on a small sample |
| P7 — Controlled experiment | Run prespecified conditions; assess answer/claim correctness, support, completeness and failure classes | Results tables + confidence intervals + error analysis | Not started | Data answers the locked RQ and includes negative results |
| P8 — Final build/report | Package code, README, reproducibility instructions, report and presentation | Final project | Not started | Claims match results; limitations and unresolved literature are disclosed |

## Current candidate decisions

| Candidate | Decision | Reason |
|---|---|---|
| C1 — generic stage-wise Medical RAG failure localization | **STOP** | Directly overlaps existing stage-aware evaluation and diagnostic approaches, including Kim et al. and StagRAG's abstract. |
| C2 — residual errors under a fixed, independently verified sufficient evidence set | **PROVISIONAL — NOT CLEARED** | Potentially narrower, but must be compared against accessible full texts including Kim et al. and CAF-RAG. StagRAG remains an acknowledged but non-blocking unresolved risk. |
| C3 — broad evidence sufficiency / abstention | **STOP in broad form** | Evidence sufficiency and abstention have already been benchmarked. |
| C4 — generic longitudinal/patient-specific RAG failure | **NOT PRIMARY / CROWDED** | Too broad; many ongoing temporal and longitudinal RAG directions. |

## Bounded Round 2 competitor set

1. **Kim et al. (2025), Rethinking Retrieval-Augmented Generation for Medicine** — open full text: https://arxiv.org/abs/2511.06738
2. **CAF-RAG (Azami et al., 2026), When does retrieval help medical question answering?** — open-access article: https://doi.org/10.1016/j.imu.2026.101810
3. **Zhao et al. (2026), Evaluation Methods for Inference-Time RAG/GraphRAG in Health Care: Scoping Review** — https://www.jmir.org/2026/1/e90046
4. **Xiong et al. (2024), MIRAGE/MEDRAG** — https://aclanthology.org/2024.findings-acl.372/
5. **Zhang & Wu (2026), Evidence Sufficiency Benchmark for Answer-Abstention Calibration** — https://doi.org/10.32604/cmc.2026.086343
6. **StagRAG (2025)** — bibliographic record/abstract only in the current access situation: https://pure.psu.edu/en/publications/stagrag-explaining-where-rag-systems-go-wrong-in-medical-reasonin/

No additional broad survey hunt unless one of these sources points to a direct experimental predecessor that could invalidate the current candidate.

## Current working research direction (not yet approved)

> Under a fixed medical QA question and a fixed evidence bundle that has been independently judged sufficient, what errors remain in model answers, and does an explicit evidence-grounding constraint reduce unsupported or incomplete claims compared with a context-only prompt?

This wording is a testable working candidate, **not a novelty claim**. It may be narrowed, changed, or rejected after the accessible competitor and dataset audit.

## Next actions — do these in order

### Action 1 — finish CAF-RAG extraction (highest priority)

Use its open article. Extract: dataset/split, models, exact context controls, how evidence relevance/sufficiency is defined, whether an oracle/gold evidence condition exists, scoring protocol, code/artifact availability, limitations, and which questions remain unanswered. Fill the matching row in the competitor matrix. Do not infer full-set sufficiency from fixed context alone.

### Action 2 — finish the relevant methods/results of Zhang & Wu

Only inspect how the five evidence levels are constructed/validated, what is held constant, and whether the paper studies errors after the evidence set is independently confirmed sufficient. This is a targeted extraction, not a full literature restart.

### Action 3 — make a bounded decision

- **STOP** if an accessible paper already tests the same fixed + independently sufficient evidence design and downstream intervention.
- **NARROW/PIVOT** if only the broad outcome is covered but a meaningful mechanism remains.
- **PROCEED PROVISIONALLY** if no inspected accessible paper demonstrates the exact design. The StagRAG unresolved risk must remain in the final limitations; novelty is not described as proven.

### Action 4 — dataset pilot before full build

Take 20–30 examples from one public dataset. For each, confirm the question, gold answer, candidate evidence, whether that evidence jointly supports the essential answer, and whether claim-level scoring is feasible. Estimate annotation time and disagreement. Stop if independent sufficiency labels cannot be defined reliably.

### Action 5 — build the minimum harness

Implement only: dataset loader, no-RAG answer, vanilla RAG answer, fixed-evidence condition, logging, and a first-pass scorer. Do not add agent/memory/graph/multimodal modules unless the RQ later requires them.

## Build Gate

The **final research build** is approved only when these are recorded:

- one research question and falsifiable hypotheses;
- accessible-competitor comparison with no known direct duplicate in inspected sources;
- explicit unresolved-literature caveat for StagRAG;
- dataset and pilot feasibility;
- baselines, outcome measures, and experiment matrix;
- stop/pivot conditions and a one-semester scope.

The exploratory harness may be developed before the gate. This avoids paralysis while preventing unsupported novelty claims.
