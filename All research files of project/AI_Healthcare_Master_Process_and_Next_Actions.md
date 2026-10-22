# AI Healthcare / Medical RAG — Master Process and Next Actions

**Version:** 1.0  
**Status date:** 10 October 2026  
**Purpose:** Keep the research process stable, record what is complete, and define objective gates before implementation.

---

## 0. Operating rules — freeze the process

1. **The phases do not change.** New papers may change a candidate's status, but they do not restart completed literature-synthesis phases.
2. **No paper hunting without a question.** Every new source must threaten, verify, or operationalize the current candidate. Add it only if it meets a stated rule below.
3. **No novelty claim from absence of evidence.** “I could not access the paper” is not evidence that the paper does not contain a method.
4. **Separate status from conclusion:** `VERIFIED`, `PARTIALLY VERIFIED`, `NOT VERIFIED`, `INACCESSIBLE`, or `CONTRADICTED`.
5. **No BUILD Gate pass while the nearest direct competitor is unresolved.** A provisional research prototype and feasibility tests may proceed, but the project must not be described as novel or clinically validated.
6. **Scope assumption:** one-semester student research project, offline evaluation/prototype—not a patient-facing clinical product. Revisit this only if the project requirements explicitly change.

## 1. The fixed end-to-end pipeline

| Phase | Work | Deliverable | Gate to proceed | Current status |
|---|---|---|---|---|
| P0. Scope & control | Define problem area, intended contribution, constraints, and decision log | Project charter + status ledger | Scope and research mode are explicit | **Mostly implicit; freeze now** |
| P1. Literature extraction (Parts A–D) | Extract papers, compare evidence, synthesize cross-paper patterns, map failure categories and candidate problems | `AI Healthcare Failure Analysis.md` and evidence tables | Sources/facts/inferences are distinguishable | **Completed for the existing corpus**; add a bounded 2026 novelty addendum, do not redo A–D |
| P2. Novelty Verification Round 1 | Search closest work and red-team C1–C4 | `Novelty Verification.md` | Eliminate broad/duplicate candidates | **Completed** |
| P3. Novelty Verification Round 2 | Check a fixed list of closest papers against the same five criteria; inspect methods/results/tables/code only where accessible | `Novelty Verification Round 2` comparison matrix | Direct overlap is either ruled in/out with evidence, or explicitly kept unresolved | **In progress; StagRAG full text inaccessible; open-paper alternatives available** |
| P4. Lock one question | Refine the surviving candidate into one operational RQ, hypotheses, variables, outcome definitions, and falsification rules | Research Question & Hypothesis sheet | Every term can be measured; no stage is conflated with another | **Not locked** |
| P5. Research design & feasibility | Audit candidate datasets, annotation burden, models, compute, baselines, metrics, protocol, ethics | Dataset/Baseline/Evaluation/Experiment Matrix | Data can support the RQ; experiment fits one semester | **Not started as a locked design** |
| P6. Baseline implementation | Reproduce simple no-RAG and normal-RAG baselines before proposing any intervention | Reproducible baseline code + output logs | Same data/prompts/versions yield traceable results | **Not started** |
| P7. Controlled experiments | Run the pre-specified conditions; annotate claims and error types; test hypotheses | Results tables, confidence intervals, error analysis | Results answer the RQ and include negative/falsifying outcomes | **Not started** |
| P8. Final project build & report | Package the evaluation prototype, README, reproducibility instructions, report, presentation | Final runnable project + report | All claims match observed evidence; limitations are explicit | **Not started** |

### Important: “Build” has two meanings

- **Allowed before the novelty gate closes:** a small exploratory harness to test dataset loading, model access, and output logging; label it *feasibility prototype*, not the final novel system.
- **Not allowed before research design is locked:** adding multiple RAG modules, agents, memory, rerankers, or a user-facing medical app without a research reason.

---

## 2. What has already been done

### Completed synthesis work

- **Part A — Literature Extraction:** the current source corpus was identified and extracted.
- **Part B — Cross-paper Evidence Table / Findings:** recurring failure patterns and limitations were compared.
- **Part C — AI Healthcare Failure Map:** failure categories and cross-paper synthesis were documented.
- **Part D — Candidate research directions:** four broad candidates were considered.
- **Novelty Verification Round 1:** broad stage-wise diagnosis was rejected; broad evidence sufficiency/abstention was rejected; longitudinal RAG was considered crowded; C2 was retained only as a provisional candidate.

These phases are **not to be repeated from scratch**. New 2026 papers belong in a dated addendum to the novelty audit unless they materially invalidate an existing synthesis claim.

### Current candidate status

| Candidate | Status | Decision rule |
|---|---|---|
| C1 — generic stage-wise Medical RAG failure localization | **STOP** | Existing work already decomposes retrieval, evidence selection, and generation; a new taxonomy alone is not enough. |
| C2 — residual errors under a fixed, independently verified sufficient evidence set | **PROVISIONAL / NOT CLEARED** | Survives only if the direct-competitor check shows a meaningful experimental difference. |
| C3 — broad evidence sufficiency / abstention | **STOP in broad form** | Existing benchmarks already evaluate evidence sufficiency and abstention. |
| C4 — generic longitudinal/patient-specific RAG failure | **CROWDED; not primary** | Keep out of the main plan unless a narrowly defined mechanism survives its own audit. |

---

## 3. Round 2: bounded competitor set

Do not keep expanding the reading list indefinitely. Use this fixed core set for the current C2 audit. Add another paper only if it directly tests fixed/oracle evidence, evidence sufficiency, evidence selection, or downstream errors.

| Priority | Paper | Access status | What it can establish | What it cannot establish by itself |
|---|---|---|---|---|
| 1 | **StagRAG** (Wang et al., IEEE BigData 2025), DOI 10.1109/BigData66926.2025.11402407 | **Full text currently inaccessible** in the checked links: GitHub URL gave 404; IEEE page gave HTTP 420 in user's browser; bibliographic record/abstract remain available | Abstract confirms stage-wise diagnosis and oracle-guided interventions; this makes it a direct threat | Abstract cannot answer whether evidence sufficiency is independently verified, exact passages are held fixed, selection is removed, or residual failures are quantified after all gates pass |
| 2 | **Kim et al. (2025), Rethinking Retrieval-Augmented Generation for Medicine** | **Open full text on arXiv**: https://arxiv.org/abs/2511.06738 | Direct overlap: retrieval → evidence selection → generation, expert annotation, claim-level factuality/completeness | The inspected methods/results use top-16 retrieved passages and measure relevance/coverage; they do not describe a controlled condition with a fixed, independently verified sufficient evidence bundle |
| 3 | **CAF-RAG (Azami et al., 2026), When does retrieval help medical question answering?** | **Open-access article**: https://doi.org/10.1016/j.imu.2026.101810; paper reports public artifacts/code at https://github.com/youssefAZAMI360/CAFRAG.git | Recent direct/adjacent threat: diagnostic comparison of six retrieval conditions, two LLMs, error taxonomy, clinical filtering, and fixed context comparisons across models | Its abstract does not establish an oracle/independently sufficient evidence condition; check full methods before deciding overlap |
| 4 | **Zhao et al. (2026), Evaluation Methods for Inference-Time RAG/GraphRAG in Health Care: Scoping Review** | **Open full text**: https://www.jmir.org/2026/1/e90046 | Landscape-level evidence about how rarely retrieval and evidence support are independently evaluated | A review-level gap is not proof that this specific C2 experiment is novel |
| 5 | **Xiong et al. (2024), Benchmarking Retrieval-Augmented Generation for Medicine / MIRAGE / MEDRAG** | **Open full text/PDF**: https://aclanthology.org/2024.findings-acl.372/ | Medical RAG benchmark, retriever/corpus/model variation, useful baseline and dataset context | End-task accuracy/component comparisons do not by themselves establish post-sufficiency residual failure attribution |
| 6 | **Zhang & Wu (2026), Evidence Sufficiency Benchmark for Answer-Abstention Calibration** | Publisher page: https://doi.org/10.32604/cmc.2026.086343 | Threatens any broad claim about evidence sufficiency and abstention; helps operationalize insufficient/conflicting evidence | Does not automatically answer whether residual errors persist after evidence has already passed an independent sufficiency test |

### What has been verified from the open alternatives so far

- **Kim et al.:** the full text describes 18 medical professionals, 80,502 annotations, 200 queries, and 800 outputs across four model settings. Its stages are retrieval relevance, evidence selection/use, and output factuality/completeness. A retrieved passage was labeled relevant if it fully or partially supported at least one “must-have” statement. The paper reports that top-16 retrieval coverage was only about 33% of must-have statements overall. That is not the same as constructing a fixed evidence bundle independently judged sufficient for each question and then measuring residual errors under that bundle.
- **MIRAGE/MEDRAG:** the open ACL paper contains 7,663 questions across five medical QA datasets and systematically varies corpora, retrievers, and LLMs. It is valuable for baselines and for showing performance can vary by task, but its main benchmark outcome is answer accuracy.
- **Zhao et al. scoping review:** among 157 studies through 14 May 2026, 47 (29.9%) reported independent retrieval-layer evaluation, 22 (14.0%) reported fine-grained evidence verification, and 116 (73.9%) reported no evidence-support verification. This supports the broader evaluation problem, not by itself a claim of novelty for C2.
- **CAF-RAG:** the September 2026 paper is a newly surfaced, important adjacent threat. Its abstract describes a failure-taxonomy/diagnostic study comparing retrieval conditions, with public per-question outputs and contexts. This paper must be read before C2 is narrowed further.
- **StagRAG:** its abstract establishes overlap with generic stage-wise failure attribution. The five finer-grained criteria below remain unresolved without the full paper or equivalent author-provided manuscript/materials.

### The five required questions — apply exactly the same definitions to each direct-threat paper

1. **Oracle evidence:** Does the experiment provide gold supporting evidence / an oracle-corrected evidence set, or only improve retrieved results?
2. **Evidence fixed:** Are the exact passages, order, prompt, and generation conditions fixed for the comparison used to make the causal claim?
3. **Independent sufficiency:** Is the *whole evidence set* independently judged sufficient to answer the question, independently of the tested model output? Relevance alone does not pass.
4. **Selection isolated:** Is learned/model-driven evidence selection removed or separately measured before downstream reasoning/generation is tested?
5. **Residual error measured:** Are incorrect/incomplete responses quantified specifically among cases where the preceding gates passed?

For every “yes,” record page/section/table/figure or a source URL. If no direct evidence is found, write `NOT REPORTED IN THE INSPECTED MATERIAL`, not `DOES NOT EXIST`.

---

## 4. Immediate next actions — in this exact order

### Action 1 — Finish the accessible-paper matrix first

**Owner:** research team  
**Scope:** Kim, CAF-RAG, Zhao review, MIRAGE, Zhang & Wu, plus StagRAG abstract-level facts.  
**Output:** one matrix with five criteria, exact supporting sections/tables, and status labels.  
**Done when:** each of the five papers is assessed only on evidence actually inspected. No new general literature browsing during this action.

### Action 2 — Make one bounded final attempt for StagRAG full text

Try only these routes: university library/institutional access; conference proceedings or an author-hosted accepted manuscript; corresponding author request. Record date and result. Do not spend repeated sessions retrying the same 404/420 pages.

**If obtained:** apply the same five criteria, citing exact pages/tables.  
**If not obtained:** freeze status as `INACCESSIBLE — DIRECT OVERLAP UNRESOLVED`. Do not pretend that other papers replace it. C2 may progress only as a **provisional design**, with the StagRAG risk disclosed; a final novelty claim remains uncleared.

### Action 3 — Rewrite C2 as an operational research question (draft only until Action 1 is done)

Working draft:

> Among medical QA items supplied with a fixed, independently verified sufficient evidence bundle, what proportion of answers remain incorrect or incomplete, and does an explicit evidence-to-claim grounding constraint reduce those errors compared with a plain-context prompt using the same evidence?

This is not yet approved as novel. It must survive comparison with StagRAG, Kim, and CAF-RAG. Do not broaden it to “why does RAG fail?”

### Action 4 — Audit datasets before committing to one

For each candidate dataset, record: public license/access; question type; ground-truth answer; supporting context/evidence availability; whether an evidence bundle can be independently judged sufficient; answer format; sample size; annotation burden; known leakage/overlap; permitted research use.

**Pass condition:** choose one primary dataset and at most one secondary validation dataset. Prefer data that already includes evidence/context. If sufficient evidence needs manual annotation, estimate the effort on a pilot of 20–30 questions first.

### Action 5 — Lock experiment conditions and metrics

Minimum candidate conditions (finalize only after dataset audit):

- **B0:** question-only / no external evidence.
- **B1:** ordinary retrieval-based RAG.
- **B2:** fixed gold/curated evidence bundle, independently judged sufficient.
- **B3:** same B2 evidence and same model/prompt settings, with a prespecified evidence-to-claim grounding constraint.

Do not call B2 “oracle sufficient” unless an independent procedure actually verifies the full evidence bundle. Keep all evidence/prompt/model/version details in logs.

Primary outcomes to consider: answer correctness, completeness against predefined required facts, unsupported-claim rate, contradiction rate, claim-level evidence support. Specify adjudication rules before testing. If human annotators are used, pilot reliability and report disagreement; do not claim clinician validation unless clinicians actually evaluate it.

### Action 6 — BUILD Gate review

Only proceed to the final research build when all boxes are checked:

- [ ] One RQ with measurable variables and falsification rule.
- [ ] Direct-threat matrix complete; StagRAG either checked from full text or formally recorded as an unresolved risk.
- [ ] Dataset access, license, evidence availability, and annotation budget confirmed.
- [ ] Baselines, model versions, prompts, and metrics specified.
- [ ] Experiment matrix and analysis plan frozen.
- [ ] Scope fits one semester and does not claim clinical deployment/safety.

If StagRAG remains inaccessible, the project can build and test the provisional experiment, but the final report must not say “first,” “novel,” or “unexplored” without additional evidence. Frame the result as a controlled replication/extension unless a defendable distinction is demonstrated.

---

## 5. How to prevent the workflow changing again

Use a status ledger. Change a candidate's status only when a new source supplies evidence that meets one of these triggers:

- **STOP:** a direct prior paper demonstrably implements the same RQ/experimental control and evaluation target.
- **NARROW:** meaningful distinction remains but needs a more precise population, mechanism, intervention, or outcome.
- **PROVISIONAL:** potential difference identified but direct verification remains incomplete.
- **CLEAR FOR DESIGN:** no direct duplicate found in the bounded search and the contribution is operationally distinct. This is not a guarantee of publication novelty.
- **PIVOT:** the remaining question is not feasible or cannot be evaluated reliably with available data/resources.

Every update must include: (1) new evidence; (2) which exact decision changes; (3) what does *not* change; (4) the next gate. Do not restart Parts A–D unless a genuine contradiction invalidates a core synthesis.

---

## 6. Current decision

**Current phase:** P3 — Novelty Verification Round 2.  
**Current focus:** verify the fixed competing-paper set, especially CAF-RAG and the open full text of Kim; preserve StagRAG as an unresolved direct threat if access fails.  
**Not yet authorized:** final RQ lock, final claim of novelty, full project BUILD.  
**Allowed now:** close the accessible-paper matrix, audit a small dataset pilot, and build a feasibility harness without claiming novelty.

**Next concrete deliverable:** `Round2_Competitor_Matrix` with rows for StagRAG, Kim, CAF-RAG, Zhao review, MIRAGE, and Zhang & Wu, and columns for the five verification questions plus the source locations that support every decision.
