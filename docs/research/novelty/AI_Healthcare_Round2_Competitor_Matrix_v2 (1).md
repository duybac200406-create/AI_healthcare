# AI Healthcare — Novelty Verification Round 2: Competitor Matrix v2

**Status date:** 15 October 2026  
**Change from previous matrix:** StagRAG remains a known unresolved direct-overlap risk but is no longer a blocker for all project work. The accessible literature is the active verification path.

## Fixed criteria

1. **Oracle/gold evidence:** Does the study supply gold/corrected supporting evidence as an intervention?
2. **Fixed evidence:** Are the exact evidence passages/bundle held constant for the relevant comparison?
3. **Independent sufficiency:** Is the complete evidence set independently judged sufficient for the target question, separate from the model answer?
4. **Selection isolated:** Is learned/model-driven evidence selection removed or measured separately before reasoning/generation is evaluated?
5. **Residual errors after gates:** Are downstream errors quantified only after the preceding controls pass?

Status vocabulary: `CONFIRMED`, `PARTIAL`, `NOT IDENTIFIED IN INSPECTED MATERIAL`, `UNVERIFIED`, `INACCESSIBLE`. “Not identified” is not proof the paper does not contain it.

## Comparison matrix

| Paper | 1. Oracle/gold evidence | 2. Exact evidence fixed | 3. Independent sufficiency | 4. Selection isolated | 5. Residual errors after all gates | Decision impact |
|---|---|---|---|---|---|---|
| **StagRAG (Wang et al., 2025)** — https://doi.org/10.1109/BigData66926.2025.11402407 | **PARTIAL:** abstract states oracle-guided interventions | **INACCESSIBLE / UNVERIFIED** | **INACCESSIBLE / UNVERIFIED** | **INACCESSIBLE / UNVERIFIED** | **INACCESSIBLE / UNVERIFIED** | Reject generic stage-wise diagnosis. Keep the narrower overlap risk unresolved, but do not block progress or keep retrying the same access routes. |
| **Kim et al. (2025)** — https://arxiv.org/abs/2511.06738 | **NOT IDENTIFIED in inspected main text:** main condition uses top-16 retrieved passages and interventions such as filtering/query reformulation | **PARTIAL:** contexts are attached to each output, but not a controlled same-bundle test of independently sufficient evidence | **NOT IDENTIFIED:** relevance is labelled against must-have statements; reported top-16 coverage is incomplete (about 33% overall) | **PARTIAL:** evidence use is directly measured, but downstream output errors are not conditioned on passing an independent sufficiency gate | **NOT IDENTIFIED** | Strong direct overlap with stage-aware evaluation, but not yet evidence of the exact proposed controlled-evidence protocol. |
| **CAF-RAG (Azami et al., 2026)** — https://doi.org/10.1016/j.imu.2026.101810 | **PARTIAL:** controlled retrieval/selection interventions, but no oracle/gold sufficient-evidence condition identified in inspected methods | **PARTIAL:** prompts/questions/settings controlled, but context varies by condition; byte-identical context is reported for a cross-model comparison, not as a post-sufficiency test | **NOT ESTABLISHED:** no independent joint evidence-sufficiency gate identified | **PARTIAL:** selection methods are compared, but not separated from generation after passing a sufficiency gate | **NOT ESTABLISHED:** no residual-error rate conditional on fixed, independently sufficient evidence | **B — PARTIAL OVERLAP; C2 MAY SURVIVE.** Strong overlap with broad retrieval-failure analysis, but not the exact C2 controlled-evidence condition in inspected material. C2 remains provisional pending Kim/Zhang & Wu and StagRAG remains unresolved. See `CAF-RAG_Full_Paper_Verification_Report.md`. |
| **Zhao et al. (2026), JMIR scoping review** — https://www.jmir.org/2026/1/e90046 | Not applicable at review level | Not applicable at review level | **Field-level fact:** 22/157 studies reported fine-grained evidence verification; 116/157 did not report evidence-support verification | **Field-level fact:** 47/157 had independent retrieval-layer evaluation | Not an experiment on C2's exact condition | Establishes evaluation practice is inconsistent, but does not establish the exact research gap. |
| **Xiong et al. (2024), MIRAGE/MEDRAG** — https://aclanthology.org/2024.findings-acl.372/ | **NOT IDENTIFIED** in the inspected main paper | **PARTIAL:** some comparisons share corpora/configuration, but broad evaluation varies retriever/corpus/backbone | **NOT IDENTIFIED:** multiple-choice accuracy is primary; no independent full-bundle sufficiency test identified | **NOT IDENTIFIED** for the C2 gate | **NOT IDENTIFIED** | Strong benchmark/baseline source; weak as a direct test of post-sufficiency residual errors. |
| **Zhang & Wu (2026)** — https://doi.org/10.32604/cmc.2026.086343 | Evidence conditions are manipulated; oracle semantics need method extraction | **PARTIAL / NEEDS METHODS CHECK** | **RELEVANT, NOT EQUIVALENT:** five-level conditions cover full/partial/irrelevant/no/conflicting evidence; verify annotation/construction details | **NOT ESTABLISHED for C2** | **NOT ESTABLISHED for C2** | Broad sufficiency/abstention direction remains stopped. Extract only details needed to evaluate overlap with C2. |

## CAF-RAG — full-paper verification completed (indexed publisher text)

**Verdict:** **B. PARTIAL OVERLAP; C2 MAY SURVIVE.** The audit inspected substantial full-text material surfaced by the official ScienceDirect indexed result (including §§1, 3.1–3.7, §5, §7 and data availability); direct page-open returned a cache-miss error and no separate local PDF was downloaded. The detailed report is `CAF-RAG_Full_Paper_Verification_Report.md`.

Five fixed criteria:

1. Oracle/gold evidence: **PARTIAL** — multiple retrieval/selection interventions, but no oracle/gold sufficient-evidence condition identified.
2. Exact evidence held fixed: **PARTIAL** — questions/prompts/settings are controlled while supplied context changes across retrieval conditions; byte-identical contexts across models do not establish a post-sufficiency fixed-bundle test.
3. Independent sufficiency: **NOT ESTABLISHED** — no independent joint-sufficiency label/gate identified.
4. Selection separated from generation: **PARTIAL** — selector methods are compared, but errors are not measured after a verified selection/sufficiency gate.
5. Residual errors after all gates pass: **NOT ESTABLISHED** — final accuracy and a retrieval-effect taxonomy are reported, not a residual-error rate conditioned on independently sufficient fixed evidence.

CAF-RAG strongly threatens generic failure analysis (C1), but does not itself establish that C2's exact post-sufficiency experimental condition has already been performed. This is a scoped judgment, not a global novelty clearance. Continue with the targeted Zhang & Wu methods audit; keep StagRAG `DIRECT OVERLAP UNRESOLVED — NON-BLOCKING`.

## CAF-RAG — source-derived facts to preserve

The open-access CAF-RAG article (September 2026) describes a diagnostic study, rather than a claim that RAG should always be used. It uses text-only pipelines with questions from SLAKE and VQA-RAD and external validation on PubMedQA (n=1,000); 1,512 questions in total; two LLMs (GPT-4o-mini and BioMistral-7B); and six conditions including no-RAG, dense RAG, hybrid BM25+dense, two rerankers, and CAF-RAG. It reports that retrieval did not significantly improve over parametric knowledge on the groups tested and degraded performance on external PubMedQA by 12 percentage points. The study also documents a substring-matching scoring artifact, and reports a corrected scorer checked against 197 blinded human gradings (91.9% agreement, kappa=0.79). Its artifacts reportedly include outputs, retrieved contexts, prompts, filter decisions, and a rebuilt 2,851-passage knowledge base.

Source: https://www.sciencedirect.com/science/article/pii/S2352914826000808

**Interpretation limit:** these are important adjacent findings, but they do not yet show that CAF-RAG fixes an independently sufficient evidence bundle and measures downstream errors conditional on sufficiency. This must be checked from the relevant methods/results rather than inferred from the abstract.

## Current decision

- **C1:** STOP (generic stage-wise failure diagnosis is too close to existing work).
- **C2:** PROVISIONAL; novelty not cleared yet. Continue using accessible papers. StagRAG remains a non-blocking unresolved risk.
- **C3:** STOP in broad form (evidence sufficiency/abstention is already directly studied).
- **C4:** Not the primary direction (crowded; requires a separate narrowly evidenced mechanism).

## The working question (not yet locked)

> Under a fixed medical QA question and a fixed evidence bundle independently judged sufficient, what errors remain in model answers, and does an explicit evidence-grounding constraint reduce unsupported or incomplete claims compared with a context-only prompt?

The novelty claim is **not** “RAG fails after retrieval” or “stage-wise RAG diagnosis is new.” The prospective contribution, if it survives the accessible competitor audit and pilot, is an experimentally controlled condition and error analysis. If a directly equivalent accessible study exists, stop or pivot.

## Required next action

1. Extract CAF-RAG methods and results against criteria 1–5; inspect any public artifacts only if accessible without spending a long time troubleshooting.
2. Extract only the sufficiency construction and controlled conditions from Zhang & Wu.
3. Review the comparison once and decide `STOP`, `NARROW/PIVOT`, or `PROCEED PROVISIONALLY`.
4. In parallel, run a small 20–30-item dataset/evidence feasibility audit. Do not wait for StagRAG to do this.
