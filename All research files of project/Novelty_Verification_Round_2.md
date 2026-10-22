# Novelty Verification — Round 2

**Audit date:** 2026-10-07  
**Purpose:** Continue the adversarial novelty audit before any Research Design / BUILD decision.  
**Scope:** Verify the strongest threats from Round 1, re-test the surviving C2 direction, and correct the weak “forward citation” claim in the previous document.

---

# 1. Round 2 Executive Verdict

## Current status

> **BUILD: NO**  
> **Research direction: C2 remains a candidate only**  
> **Novelty confidence: low-to-medium until StagRAG is fully verified**

Round 2 strengthens the case that the broad C2 phenomenon is **already established**:

> relevant medical evidence can be retrieved, yet the model may still fail to use it correctly or produce a correct answer.

The remaining possible novelty is narrower:

> **a controlled causal study in which sufficient evidence is independently verified, held fixed, and retrieval/evidence-selection variance is removed before measuring residual reasoning/generation failures.**

This remains potentially distinct, but it is **not yet safe to call a research gap** because StagRAG explicitly reports oracle-guided interventions and generator instability, and its full methods/results have not yet been independently inspected in this Round 2 environment.

---

# 2. Important Correction to Round 1

## E10 was not a true forward-citation audit

The previous document called a section “Forward Citation Findings,” but the evidence there was mainly:

- concept search;
- direct title search;
- discovery of later adjacent work.

That is **not equivalent to an exhaustive forward-citation graph**.

Therefore the Round 1 statement should be downgraded to:

> **“Forward-citation / follow-up search is incomplete; concept-based follow-up evidence was found.”**

This matters because:

> “No forward citation found” ≠ “no follow-up work exists.”

The corrected standard in Round 2 is:

1. verify the original paper metadata;
2. inspect its full text where possible;
3. identify explicit later papers that cite or build on it;
4. separately search adjacent work by concept;
5. never merge steps 3 and 4 into a single bibliometric claim.

---

# 3. Threat Verification — Kim et al. 2025

## Paper

**Kim et al. (2025), “Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights,” arXiv:2511.06738.**

Source: https://arxiv.org/abs/2511.06738

## Full-text verification status

**Verified from the full HTML paper.**

## Critical findings

Kim et al. explicitly decompose medical RAG into:

1. evidence retrieval;
2. evidence selection;
3. response generation.

They evaluate 18 medical experts, 200 queries, 800 model responses, GPT-4o and Llama-3.1-8B, and more than 80,000 expert annotations.

More importantly for C2, the paper does **not** stop at end-answer accuracy.

It explicitly reports:

- retrieval relevance and coverage;
- evidence-selection precision/recall;
- statement-level factuality;
- response-level and statement-level completeness;
- cases where relevant evidence was retrieved but not cited/used.

Their evidence-retrieval definition is already clinically functional: a passage is relevant when it partially or fully supports at least one “must-have” statement.

At k=16 they report approximately 22% relevant retrieved passages and about 33% must-have statement coverage. Evidence selection is also weak: GPT-4o precision 41.2%, recall 48.6%; Llama-3.1 precision 43.0%, recall 27.5%.

The paper additionally analyzes statements supported by relevant retrieved evidence, statements tied to irrelevant evidence, self-generated evidence, and statements without references.

### Consequence for C2

The broad claim:

> “Medical RAG can fail despite relevant evidence being retrieved.”

is **not novel**.

Kim et al. already directly document the phenomenon.

### What Kim does NOT cleanly establish

Kim’s study does **not** fully remove retrieval/evidence-selection uncertainty before assessing downstream reasoning/generation. Their empirical pipeline is observational over a live RAG process, even though it is stage-wise annotated.

Therefore the residual candidate remains narrower than Kim:

> **Hold a verified sufficient evidence set constant, then test whether the same evidence still produces materially different correctness/completeness outcomes because of downstream interpretation/integration/reasoning/generation.**

That distinction is potentially real, but it is a **methodological distinction**, not yet a proven novelty claim.

---

# 4. Threat Verification — StagRAG 2025

## Paper

**Wang et al. (2025), “StagRAG: Explaining Where RAG Systems Go Wrong in Medical Reasoning,” IEEE BigData 2025, pp. 8030–8039.**

DOI: https://doi.org/10.1109/BigData66926.2025.11402407

Publisher/indexing evidence confirms publication in IEEE BigData 2025.

## Verified facts available in Round 2

The publisher record explicitly states that StagRAG:

- is a stage-wise diagnostic framework for medical RAG;
- attributes errors to retriever, reranker, or generator;
- uses oracle-guided interventions;
- uses auxiliary inference probes;
- reports insufficient retrieval coverage and generator instability as important failure sources.

## Critical limitation of the current audit

The complete IEEE paper was not retrievable from the available web interface during this round, so the following questions remain **OPEN**:

1. What exactly is the oracle intervention?
2. Does the oracle replace the entire retrieved context with gold/sufficient evidence?
3. Does it explicitly control evidence sufficiency, or only retrieval/reranking quality?
4. How is “generator instability” operationalized?
5. Can a failure be classified as downstream reasoning/generation while evidence is held constant?
6. What datasets, models, and sample sizes are used for the generator-stage attribution?
7. Does the paper measure the same evidence repeated across multiple generation conditions?
8. Are residual failures categorized beyond a generic generator label?

## Round 2 verdict

> **StagRAG is the blocking paper for closing C2 novelty.**

Until these eight questions are answered from the full paper, it is not defensible to claim that the proposed controlled residual-error study is genuinely new.

This is the most important unresolved item in Round 2.

---

# 5. Threat Verification — Evidence Sufficiency / Abstention

## Zhang & Wu 2026

**“Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in Retrieval-Augmented Generation.”**

DOI: https://doi.org/10.32604/cmc.2026.086343

The published article introduces a five-level evidence-sufficiency benchmark:

- L1 Full Support
- L2 Partial Support
- L3 Irrelevant Evidence
- L4 No Context
- L5 Conflicting Evidence

It evaluates seven LLMs and studies answer/abstention behaviour under different evidence conditions.

### Consequence

The broad research problem:

> “Can LLMs recognize when evidence is insufficient?”

is already substantially benchmarked.

Therefore C3 should remain **STOP in broad form**.

This work does not directly invalidate the narrow C2 question because C2 conditions on **sufficient** evidence rather than studying abstention under insufficient evidence.

---

# 6. Threat Verification — Dadi 2026

## Paper

**Divya Dadi (2026), “Evaluating Evidence Verification and Selective Abstention for Reliable Retrieval-Augmented Biomedical Question Answering.”**

SSRN posting: 2 Sep 2026.  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7391678

The study uses PubMedQA with retrieval confidence, evidence verification, and selective abstention.

### Consequence

Biomedical selective abstention is no longer an untouched direction.

Again, this does not directly kill the C2 residual-reasoning hypothesis, but it further removes the possibility of claiming novelty from “evidence-aware reliability” or “abstention” in generic form.

---

# 7. New 2026 Threat — ER-MedRAG

## Paper

**ER-MedRAG: A multi-agent reinforcement learning framework for reliable clinical retrieval-augmented reasoning.**

Published 16 May 2026.

https://link.springer.com/article/10.1007/s44443-026-00825-0

## Why it matters for C2

ER-MedRAG explicitly frames a key problem as:

> even when relevant material is retrieved, the model may not effectively transform that evidence into reliable reasoning.

It introduces an Extractor–Respondent architecture in which:

- retrieved snippets are converted into structured condition–relation–conclusion reasoning triplets;
- the respondent is trained to use those triplets;
- downstream answer correctness is used in the training signal;
- ablations test the effect of the reasoning representation.

### Consequence

This means that the broad mechanism:

> “evidence → reasoning integration is a failure point”

is already actively studied in 2026.

However, ER-MedRAG is primarily an **intervention / optimization framework**, not a controlled failure-attribution study asking:

> with evidence fixed and verified as sufficient, what residual mechanisms explain the remaining errors?

So it **narrows** C2 but does not automatically eliminate it.

---

# 8. New 2026 Scoping Review — Critical Landscape-Level Evidence

## Paper

**Zhao et al. (2026), “Evaluation Methods for Inference-Time Retrieval-Augmented and Graph Retrieval-Augmented Large Language Models in Health Care: Scoping Review.”**

JMIR 2026;28:e90046.  
https://www.jmir.org/2026/1/e90046

Searches covered PubMed, Web of Science, IEEE Xplore, ACM Digital Library, arXiv and medRxiv through 14 May 2026, including backward and forward citation tracking.

The review included **157 studies**.

## Key findings

- independent retrieval-layer evaluation: 47/157 (29.9%);
- grounding/faithfulness evaluation: 41/157 (26.1%);
- fine-grained evidence verification: 22/157 (14.0%);
- claim/statement-level verification: 17/157 (10.8%);
- formal safety evaluation: 45/157 (28.7%);
- human evaluation: 94/157 (59.9%).

The review explicitly states that when retrieval is evaluated only through final outputs, errors become difficult to attribute to retrieval, evidence selection, or generation.

### Consequence for C2

This review supports the existence of a **real evaluation-method problem**:

> layer-specific evaluation is inconsistent and end-to-end scores make error attribution difficult.

But it does **not** prove that our exact controlled-sufficient-evidence RQ is novel.

Instead, it provides a strong reason to keep investigating the experimental design while remaining cautious about the novelty claim.

---

# 9. Longitudinal C4 Re-check

## EHR-RAG 2026

EHR-RAG explicitly targets long-horizon structured EHRs and introduces:

- event- and time-aware hybrid retrieval;
- adaptive iterative retrieval;
- dual-path evidence retrieval and reasoning.

It reports gains across four long-horizon EHR prediction tasks.

Source: https://arxiv.org/abs/2601.21340

## LongMedBench 2026

LongMedBench provides a real-world EHR benchmark for long-horizon clinical decision-making and reports an important separation:

> longer context, RAG, and agent memory help factual retrieval more than long-horizon decision-making.

Source: https://papers.miccai.org/miccai-2026/0593-Paper4292.html

## MedEventGraph-RAG 2026

This work directly targets longitudinal evidence verification under:

- temporal mismatch;
- negation;
- repeated documentation;
- conflicting findings;
- evidence insufficiency.

Source: https://arxiv.org/abs/2608.22062

## Jung et al. 2026

A clinical RAG conversational experiment reported hallucination increasing from 5% with no history to 40% after 10 prior exchanges.

Source: https://www.nature.com/articles/s44401-026-00115-x

### C4 verdict

C4 remains **highly crowded**.

A generic “longitudinal RAG failure” project is not defensible as novel.

---

# 10. Updated Candidate Overlap Matrix

| Candidate | Strongest overlap | What is already established? | Remaining possible distinction | Round 2 verdict |
|---|---|---|---|---|
| C1 Stage-wise failure localization | Kim 2025; StagRAG 2025 | Retrieval/selection/generation attribution already exists | Universal taxonomy | **STOP** |
| C2 Residual failure after sufficient evidence | Kim 2025; StagRAG 2025; ER-MedRAG 2026 | Relevant evidence can be unused; downstream reasoning/use is a known problem | Controlled fixed-evidence causal isolation | **CANDIDATE ONLY** |
| C3 Evidence sufficiency/abstention | Zhang & Wu 2026; Dadi 2026 | Benchmarking + biomedical selective abstention already exist | New medical mechanism only | **STOP / PIVOT** |
| C4 Longitudinal RAG failure | EHR-RAG; LongMedBench; MedEventGraph-RAG; Jung 2026 | Long-horizon retrieval, memory, temporal conflict, conversational drift already studied | Very narrow mechanism | **NARROW / PIVOT** |

---

# 11. C2 Red-Team — Round 2

## Claim under test

> “After retrieval and evidence sufficiency are controlled, residual Medical RAG errors can be causally attributed to reasoning/generation mechanisms.”

## Strongest objection #1

> **Kim already evaluates relevant evidence that is not used.**

Correct.

Therefore the project cannot claim novelty from merely observing this phenomenon.

## Strongest objection #2

> **StagRAG already uses oracle-guided interventions and diagnoses generator instability.**

Potentially fatal.

This objection cannot yet be answered because the exact oracle design has not been verified from the full IEEE paper.

## Strongest objection #3

> **ER-MedRAG already treats evidence-to-reasoning transformation as a core failure mechanism.**

Correct.

Therefore the novelty cannot be:

> “RAG struggles to reason over retrieved evidence.”

## Strongest objection #4

> “You have simply renamed evidence-selection failure as reasoning failure.”

This remains the central methodological threat.

## Required defense

The study must make the stages empirically non-overlapping:

1. **Retrieval correctness** — Did the system obtain the target evidence?
2. **Evidence sufficiency** — Does the provided evidence jointly entail enough information for the target answer?
3. **Evidence selection/use** — Did the model actually rely on the admissible evidence?
4. **Reasoning transformation** — Did the model infer the correct intermediate conclusion from that evidence?
5. **Generation** — Did the final language correctly realize the inferred conclusion?

If stages 3–5 cannot be operationally separated, the C2 novelty claim should be abandoned.

---

# 12. The Most Defensible C2 Reformulation

The earlier RQ was:

> When medical evidence is independently verified as relevant and sufficient, what residual reasoning or generation failure modes cause a RAG system to produce incorrect or incomplete clinical answers, and can those failures be reliably attributed under controlled evidence conditions?

### Round 2 improved version

> **Under a fixed, independently verified sufficient evidence set, what residual failure mechanisms in medical question answering cause incorrect or incomplete outputs, and can controlled interventions distinguish evidence-use failure from downstream reasoning/generation failure?**

This wording is safer because it does not assume that every residual error is already “reasoning failure.”

Instead, it asks whether controlled intervention can distinguish competing explanations.

---

# 13. Stronger Experimental Logic

The minimum viable experiment should not be a normal RAG benchmark.

It should be a **controlled evidence intervention study**.

## Condition A — No external evidence

Measures baseline model performance.

## Condition B — Natural retrieved evidence

Measures ordinary RAG performance.

## Condition C — Gold / sufficient evidence

Retrieval uncertainty is removed.

## Condition D — Gold evidence + evidence-use constraint

Selection/use uncertainty is reduced further.

## Condition E — Same evidence, multiple reasoning/prompt controls

Tests whether output variability persists under identical evidence.

### Scientific comparison

The strongest contrast is not:

> RAG vs no RAG.

It is:

> **same question + same sufficient evidence + same model + changed reasoning/use intervention → changed outcome**

That comparison has a much stronger causal interpretation.

---

# 14. Falsification Rules — Keep These

The direction should be abandoned or redesigned if any of the following occurs:

1. Supplying sufficient evidence removes almost all errors.
2. Most residual errors are explained by evidence selection.
3. StagRAG already performs the same oracle-fixed experiment.
4. Annotators cannot reliably distinguish sufficiency from relevance.
5. Inter-rater agreement is weak.
6. Findings disappear across models.
7. Sample size is too small for stable error-rate estimation.
8. The final study becomes only a model comparison.

These are not failure of the research process. They are valid negative results.

---

# 15. Round 2 Build Gate

## Problem

**Not yet locked.**

Candidate:

> residual Medical RAG errors after evidence conditions are controlled.

## Research question

**Promising but not validated.**

## Hypothesis

**Testable, but high-risk and falsifiable.**

## Contribution

Potentially:

> a controlled empirical attribution protocol / empirical finding.

Not:

> a new generic RAG architecture.

## Dataset

Potentially feasible with public medical QA data, but evidence-level supervision is required.

## Baselines

Not yet locked.

## Evaluation

Needs stage-level operational definitions before implementation.

## Novelty

**Not cleared.**

## Overall gate

> **DO NOT BUILD YET**

---

# 16. Exact Blockers Remaining

## Blocker A — StagRAG full-paper verification

Highest priority.

Need to answer the 8 questions in Section 4.

## Blocker B — Actual citation-graph completion

The previous document's forward-citation section must not be treated as complete.

Need a source that provides an actual citing-paper set for the core threats, especially Kim and StagRAG.

## Blocker C — Operational definition of “sufficient evidence”

“Sufficient” cannot simply mean “relevant.”

A defensible label should mean that the provided evidence jointly contains enough information to derive the target answer under the task definition.

## Blocker D — Evidence-use vs reasoning separation

This is the hardest methodological problem.

A model can have sufficient evidence and still fail because:

- it ignores the evidence;
- it misreads the evidence;
- it combines it incorrectly;
- it reaches the wrong intermediate conclusion;
- it verbalizes a correct internal conclusion incorrectly.

The experiment must distinguish these possibilities.

---

# 17. Final Round 2 Status

## What Round 2 has established

**Established with high confidence:**

- C1 broad novelty is not defensible.
- C3 broad novelty is not defensible.
- C4 broad novelty is not defensible.
- Kim et al. directly establish relevant-evidence / evidence-use failures.
- 2026 research actively targets evidence-to-reasoning integration.
- 2026 healthcare RAG evaluation literature still shows major inconsistency in layer-specific evaluation.

**Still unresolved:**

> whether a rigorously controlled, fixed-sufficient-evidence experiment that attributes residual errors beyond evidence selection is sufficiently different from StagRAG and related work to justify a novel research contribution.

## Current decision

> **C2 = CANDIDATE, NOT CLEARED**

> **BUILD = NO**

> **Next action = full StagRAG verification + final C2 novelty test**

---

# 18. Source List Used in Round 2

1. Kim et al. 2025 — Rethinking Retrieval-Augmented Generation for Medicine  
   https://arxiv.org/abs/2511.06738

2. Wang et al. 2025 — StagRAG  
   https://doi.org/10.1109/BigData66926.2025.11402407

3. Zhang & Wu 2026 — Evidence Sufficiency Benchmark  
   https://doi.org/10.32604/cmc.2026.086343

4. Dadi 2026 — Evidence Verification and Selective Abstention  
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7391678

5. Shi et al. 2026 — ER-MedRAG  
   https://doi.org/10.1007/s44443-026-00825-0

6. Zhao et al. 2026 — Evaluation Methods for Inference-Time RAG/GraphRAG in Health Care: Scoping Review  
   https://doi.org/10.2196/90046

7. Cao et al. 2026 — EHR-RAG  
   https://arxiv.org/abs/2601.21340

8. Chen et al. 2026 — LongMedBench  
   https://papers.miccai.org/miccai-2026/0593-Paper4292.html

9. Lin et al. 2026 — MedEventGraph-RAG  
   https://arxiv.org/abs/2608.22062

10. Jung et al. 2026 — RAG in clinical practice: a cautionary tale of AI “Truthfulness”  
    https://doi.org/10.1038/s44401-026-00115-x

---

## Audit principle

> **Do not promote C2 from “candidate” to “novel” until the closest prior work has been shown to ask a materially different question or use a materially different causal identification protocol.**
