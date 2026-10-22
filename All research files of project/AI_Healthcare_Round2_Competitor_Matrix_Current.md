# AI Healthcare — Novelty Verification Round 2: Competitor Matrix (Working Draft)

**As of:** 10 October 2026  
**Purpose:** Apply the same five tests to the same bounded competitor set. This is a working matrix, not a final novelty clearance.

## Status legend

- **CONFIRMED** — directly supported by inspected source material.
- **PARTIAL** — related measurement/intervention exists, but not the exact required control.
- **NOT IDENTIFIED IN INSPECTED MATERIAL** — the examined text does not describe it; do not infer it does not exist elsewhere.
- **UNVERIFIED / INACCESSIBLE** — evidence needed to decide is not currently accessible.

## Five fixed criteria

1. **Oracle evidence:** Was gold/corrected supporting evidence supplied as an intervention?
2. **Fixed evidence:** Were the exact passages held constant in the experiment relevant to the claim?
3. **Independent sufficiency:** Was the full evidence set independently judged sufficient, separate from the tested model's answer?
4. **Selection isolated:** Was model-driven evidence selection removed or separately assessed before downstream reasoning/generation was tested?
5. **Residual errors measured:** Were errors quantified only after the previous conditions passed?

## Matrix

| Paper | 1. Oracle evidence | 2. Exact evidence fixed | 3. Sufficiency independently verified | 4. Selection isolated | 5. Residual error after all gates | Interim implication for C2 |
|---|---|---|---|---|---|---|
| **StagRAG (2025)** | **PARTIAL:** abstract confirms “oracle-guided interventions”; the supplied object/implementation of the oracle is unknown | **UNVERIFIED / INACCESSIBLE** | **UNVERIFIED / INACCESSIBLE** | **UNVERIFIED / INACCESSIBLE** | **UNVERIFIED / INACCESSIBLE** | **Direct threat remains unresolved.** Abstract is enough to reject generic stage-wise diagnosis, but not enough to decide the narrowed C2 protocol. GitHub URL shown by user returns 404; IEEE Xplore returned HTTP 420 in user's browser. These access failures are not evidence of absence in the paper. |
| **Kim et al. (2025), Rethinking RAG for Medicine** | **NOT IDENTIFIED IN INSPECTED MAIN TEXT:** main method uses retrieved top-16 passages and evaluates filtering/query reformulation; no gold-evidence oracle condition was identified in the inspected methods/results | **PARTIAL:** passage contexts are known for each RAG output; the main study does not establish a controlled comparison using the same independently sufficient evidence bundle | **NOT IDENTIFIED:** passage relevance is defined as fully/partially supporting at least one must-have statement; that is not full-set sufficiency. The reported top-16 retrieval coverage was about 33% of must-have statements overall | **PARTIAL:** evidence selection/use is explicitly evaluated (precision/recall via cited/retrieved passage alignment), but downstream error is not reported only on cases where selection passed | **NOT IDENTIFIED:** factuality/completeness are measured across model conditions; no tested subset conditioned on independently sufficient evidence and selection success was identified | **Strong overlap with stage-wise evaluation. A possible distinction remains around an independently verified sufficient evidence bundle and conditioned residual error, but this is not yet a validated gap.** |
| **CAF-RAG (Azami et al., 2026)** | **UNVERIFIED pending methods check:** the abstract describes six retrieval conditions, including No-RAG and rerank/filter variants; no oracle-sufficient evidence condition is described in the abstract | **PARTIAL:** abstract says GPT-4o-mini and BioMistral-7B received byte-identical retrieved context in the relevant cross-model comparison | **NOT IDENTIFIED IN ABSTRACT:** context is retrieved text, not described as independently sufficient per question | **PARTIAL:** passage-selection/filtering is central to the study, but the exact gate and whether selection uncertainty is removed need full methods inspection | **NOT IDENTIFIED IN ABSTRACT:** residual errors after an independently verified sufficient set are not described | **Important recent adjacent threat; full methods/results and public artifacts must be checked before final C2 wording.** |
| **Zhao et al. (2026), JMIR scoping review** | Not applicable at review level | Not applicable at review level | **CONFIRMED AS A LANDSCAPE FINDING:** 116/157 studies did not report evidence-support verification; only 22/157 reported fine-grained evidence verification | **CONFIRMED AS A LANDSCAPE FINDING:** independent retrieval-layer evaluation was reported in 47/157 studies | Not a study of the exact residual-error intervention | Supports the broad need for better evaluation, but **does not establish C2 novelty**. A literature-level evaluation gap is not itself proof of an experiment-level gap. |
| **Xiong et al. (2024), MIRAGE/MEDRAG** | **NOT IDENTIFIED in inspected main paper:** main setup uses retrieved corpora/snippets, not an oracle-sufficient evidence intervention | **PARTIAL:** same MedCorp + RRF-4 + 32 retrieved snippets are used in one cross-backbone comparison; other conditions vary retrievers/corpora | **NOT IDENTIFIED:** main task measures multiple-choice QA accuracy and does not independently label complete evidence sets as sufficient | **NOT IDENTIFIED:** retrieves/concatenates snippets and evaluates end-task accuracy; no separate model-selection pass gate identified | **NOT IDENTIFIED:** no residual error subset conditioned on evidence sufficiency/selection success identified | Useful open baseline/benchmark source; not a direct substitute for StagRAG's full paper. |
| **Zhang & Wu (2026), evidence sufficiency / abstention benchmark** | **Not the central issue:** the paper manipulates evidence-level conditions; exact oracle construction should be taken from its methods if used | Evidence is condition-controlled at benchmark level, but not yet checked for the specific question of a fixed sufficient medical evidence set followed by residual-generation analysis | **PARTIAL / RELEVANT:** explicitly tests full, partial, irrelevant, no-context, and conflicting evidence conditions; verify exact annotation/construction procedure before relying on finer claims | **Not established for C2:** primarily threatens a broad sufficiency/abstention question | **Not established for C2:** broad answer/abstention outcomes are not automatically residual error attribution after selection passes | Broad C3 remains STOP; for C2, read only the methods/results relevant to how sufficiency is constructed and validated. |

## Source links and evidence notes

### 1. StagRAG

- Penn State bibliographic record and abstract: https://pure.psu.edu/en/publications/stagrag-explaining-where-rag-systems-go-wrong-in-medical-reasonin/
- DOI: https://doi.org/10.1109/BigData66926.2025.11402407
- Author-linked repository shown in the record: https://github.com/YCWang0709/StagRAG.git
- What is confirmed: stage-wise diagnosis of retriever, reranker, generator; oracle-guided interventions; auxiliary inference probes.
- What remains unverified: the exact oracle object; whether evidence is kept fixed; independent sufficiency; selection removed; residual-error rate after all gates pass.

### 2. Kim et al. (2025)

- Full text: https://arxiv.org/abs/2511.06738
- Inspected methods/results describe 18 medical experts, 80,502 annotations, 200 queries, 800 outputs, and three evaluated stages: retrieval relevance, evidence selection/use, and response generation.
- The relevance test labels a passage relevant if it fully or partially supports at least one must-have statement. The main pipeline's top-16 passages covered about 33% of must-have statements overall. This is not an independently sufficient evidence bundle.

### 3. CAF-RAG (2026)

- Open-access article: https://doi.org/10.1016/j.imu.2026.101810
- Title: *When does retrieval help medical question answering? A question-type sensitivity analysis and failure taxonomy for text-based (image-free) medical RAG, with an interpretable clinical filter (CAF-RAG)*.
- Abstract describes six retrieval conditions across two models and an external PubMedQA validation set of 1,000 questions; public artifacts include outputs, contexts, filter decisions, and prompts.
- This is an important new adjacent competitor and must not be skipped merely because StagRAG is inaccessible.

### 4. Zhao et al. (2026)

- Open-access full text: https://www.jmir.org/2026/1/e90046
- Scoping review searched through 14 May 2026 and included 157 studies. Reported: independent retrieval-layer evaluation 47/157 (29.9%); fine-grained evidence verification 22/157 (14.0%); no evidence-support verification 116/157 (73.9%).
- These are descriptive field-level counts. They do not demonstrate that the exact C2 experiment has not been conducted.

### 5. Xiong et al. (2024)

- Full paper page/PDF: https://aclanthology.org/2024.findings-acl.372/
- MIRAGE has 7,663 questions across five medical QA datasets; MEDRAG compares corpora, retrievers, and LLMs. In the backbone comparison, the study uses a shared MedCorp/RRF-4 setup and 32 snippets; its main metric is multiple-choice accuracy.

### 6. Zhang & Wu (2026)

- Publisher DOI: https://doi.org/10.32604/cmc.2026.086343
- Use it for the evidence-sufficiency/abstention threat, not as automatic proof that post-sufficiency residual attribution is already solved. Read method details before making a stronger claim.

## Current decision

**C2 = PROVISIONAL / NOVELTY NOT CLEARED.**

What is already rejected: “stage-wise Medical RAG failure diagnosis” as a generic contribution.

What remains a candidate, not a claim:

> Measure what residual errors persist when the exact evidence bundle is independently judged sufficient and held constant, and distinguish those errors from retrieval and evidence-selection failure.

## Next work item

1. Read and extract the **CAF-RAG full text** against the five criteria.
2. Finish the relevant methods/results extraction from **Zhang & Wu** for evidence-sufficiency construction.
3. Make one bounded attempt for StagRAG via institutional library / author manuscript / corresponding author. If unavailable, freeze its status as `INACCESSIBLE — DIRECT OVERLAP UNRESOLVED` rather than repeatedly trying the same pages.
4. Only then draft the final C2 RQ and audit dataset feasibility.
