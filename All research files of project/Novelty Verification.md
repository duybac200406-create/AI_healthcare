# Novelty Verification — Research State

**Audit date:** 2026-10-07  
**Research area:** Medical RAG / AI Healthcare Failure Analysis  
**Status:** Research due-diligence audit — **not** a project proposal

> **Purpose:** determine whether the current research candidates are already done, too close to prior work, incremental, meaningfully distinct, or genuinely promising.

This document deliberately uses an adversarial standard. A negative verdict is a valid research outcome.

---

# Executive Summary

## Bottom-line verdict

The original four candidates should **not** be treated as four open research gaps.

| Candidate | Current formulation | Preliminary verdict | Novelty confidence |
|---|---|---|---:|
| C1 | Stage-wise Medical RAG failure analysis | **STOP as originally formulated** | 95/100 |
| C2 | Correct retrieval → incorrect/incomplete generation | **NARROW** | 78/100 |
| C3 | Evidence sufficiency / abstention | **STOP as broad formulation; PIVOT if medically specialized** | 92/100 |
| C4 | Longitudinal / patient-specific RAG failure | **NARROW / PIVOT** | 68/100 |

### The strongest threats

1. **Kim et al., “Rethinking Retrieval-Augmented Generation for Medicine” (2025)** already performs a large-scale, expert-driven decomposition of Medical RAG into:
   - evidence retrieval;
   - evidence selection;
   - response generation;
   and explicitly studies failure attribution across these stages.

2. **Wang et al., StagRAG (IEEE BigData 2025)** explicitly proposes a stage-wise diagnostic framework that attributes medical RAG errors to the retriever, reranker, or generator.

3. **Zhang & Wu (2026)** already introduces a benchmark for evidence sufficiency and abstention calibration across five evidence conditions, including partial, irrelevant, absent, and conflicting evidence.

4. **Dadi (2026)** directly studies selective abstention for biomedical RAG using retrieval confidence and evidence verification.

5. **Cao et al., EHR-RAG (2026)** explicitly addresses long-horizon structured EHR retrieval, temporal dynamics, evidence coverage, and reasoning.

6. **Lin et al., MedEventGraph-RAG (2026)** goes further into longitudinal clinical-event verification with supporting/contradicting evidence, temporal mismatch, patient-specific graphs, and insufficient outcomes.

7. **Jung et al. (2026)** demonstrates that conversational history itself can create a clinical RAG failure mode: hallucination increased as prior dialogue accumulated.

Therefore:

> **The research space is not empty. The original ideas are already represented in the literature.**

The potentially defensible opportunity is not “analyse Medical RAG failures stage by stage”, “study correct retrieval but wrong generation”, “study evidence sufficiency”, or “study longitudinal RAG” in isolation.

The remaining opportunity must be a **more sharply controlled research question** that isolates a mechanism not adequately separated by the existing work.

---

# E0 — Research Candidates Under Verification

| Candidate | Current formulation | Novelty status |
|---|---|---|
| C1 | Stage-wise Medical RAG failure analysis | Unknown → **High overlap** |
| C2 | Correct retrieval but incorrect/incomplete generation | Unknown → **Strongly related; requires narrowing** |
| C3 | Evidence sufficiency / abstention | Unknown → **Already actively benchmarked** |
| C4 | Longitudinal patient-specific RAG failure | Unknown → **Active and rapidly expanding area** |

## Critical interpretation

The four candidates are **research areas**, not yet research questions.

That distinction matters.

For example:

> “Medical RAG failure analysis”

is a topic.

Whereas:

> “Under controlled conditions where retrieved passages are independently judged sufficient and relevant, how often does the generator produce unsupported or incomplete medical claims, and which evidence-selection properties predict those failures?”

is a research question.

The novelty audit must operate at the second level.

---

# E1 — Literature Search Strategy

## Search scope

The search was extended through **2026-10-07**, rather than stopping at 2024 or 2025.

Search families included:

### Medical RAG failure

- "medical RAG" "failure analysis"
- "medical RAG" "error analysis"
- "medical RAG" "failure taxonomy"
- "medical RAG" "error attribution"
- "clinical RAG" "failure analysis"
- "healthcare RAG" "error analysis"

### Stage-wise analysis

- "medical RAG" "stage-wise evaluation"
- "medical RAG" "component-wise evaluation"
- "medical RAG" retrieval generation failure
- "medical RAG" retriever generator error
- "medical RAG" retrieval reasoning generation
- "RAG diagnostic framework" medical

### Evidence sufficiency / abstention

- "medical RAG" "evidence sufficiency"
- "medical RAG" "insufficient evidence"
- "medical QA" abstention
- "medical RAG" abstention
- "medical question answering" uncertainty evidence
- "RAG" "conflicting evidence" medicine

### Correct retrieval / wrong generation

- "medical RAG" "correct retrieval" generation
- "retrieved evidence" "incorrect answer" medical RAG
- "retrieval success" "generation failure" RAG
- "relevant evidence" "wrong answer" RAG medical
- "grounded evidence" hallucination generation medical

### Longitudinal / patient-specific

- "longitudinal medical RAG"
- "longitudinal patient RAG"
- "patient-specific RAG" healthcare
- "EHR RAG" failure
- "EHR RAG" hallucination
- "EHR RAG" longitudinal
- "patient memory" medical AI RAG
- "temporal RAG" healthcare
- "temporal consistency" medical RAG
- "patient timeline" RAG

## Sources prioritized

1. Publisher / proceedings pages
2. Primary papers
3. arXiv where it is the primary public version
4. PubMed / biomedical databases
5. ACL / PMLR / IEEE metadata
6. Citation indexes / scholarly search
7. Secondary sources only for discovery

---

# E2 — Closest Prior Work

The following papers were selected because they threaten the candidates scientifically, not merely because they contain the word “RAG”.

---

## Prior Work #1 — Rethinking Retrieval-Augmented Generation for Medicine

- **Title:** Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights
- **Authors:** Hyunjae Kim et al.
- **Year:** 2025
- **Venue/status:** arXiv preprint, arXiv:2511.06738
- **DOI:** 10.48550/arXiv.2511.06738
- **URL:** https://arxiv.org/abs/2511.06738
- **Primary / Secondary:** Primary research
- **Evidence strength:** **Very high**
- **Candidate threatened:** C1, C2, partially C3

### Why this paper is close

This is arguably the strongest threat to the original research direction.

The authors explicitly decompose Medical RAG into:

1. evidence retrieval;
2. evidence selection;
3. response generation.

They use 18 medical experts, 80,502 annotations, 800 model outputs, 200 medical queries, and compare GPT-4o/Llama-3.1 with and without RAG.

The paper does not merely report final answer accuracy. It traces:

> retrieval → evidence selection → generation

and measures relevance, evidence usage, factuality, and completeness.

### Source facts

- Only approximately 22% of top-16 retrieved passages were judged relevant.
- Evidence selection precision was approximately 41–43%.
- Evidence selection recall was approximately 27–49%.
- RAG reduced factuality and completeness in some settings.
- The authors explicitly describe their framework as stage-aware and discuss error attribution.

### Important implication for C1

A proposal saying:

> “We will build a framework that identifies whether Medical RAG failed during retrieval, evidence selection, or generation.”

would be **very close** to this paper.

That is not a defensible novelty claim without a substantially different research question.

### Important implication for C2

The paper also provides direct evidence for:

> relevant evidence is retrieved, but the model may fail to select/use it correctly.

This is close to the proposed “correct retrieval → wrong generation” pattern.

### Our inference

C1 is effectively pre-empted in its broad formulation.

C2 remains potentially interesting only if it isolates a mechanism that this paper does not identify or control experimentally.

---

## Prior Work #2 — StagRAG

- **Title:** StagRAG: Explaining Where RAG Systems Go Wrong in Medical Reasoning
- **Authors:** Yucheng Wang, Md AlMahfuz Chowdhury, Mahmuda Akter Sristy, Md Faisal Kabir
- **Year:** 2025
- **Venue:** IEEE International Conference on Big Data 2025
- **Status:** **Published conference paper**
- **Pages:** 8030–8039
- **DOI:** 10.1109/BigData66926.2025.11402407
- **URL:** https://doi.org/10.1109/BigData66926.2025.11402407
- **Primary / Secondary:** Primary research
- **Evidence strength:** **Very high**
- **Candidate threatened:** C1, C2

### Why this paper is close

StagRAG explicitly proposes a:

> Stage-wise RAG diagnostic framework

for medical reasoning.

It attributes errors to:

- retriever;
- reranker;
- generator.

It uses oracle-guided interventions and auxiliary inference probes to localize error sources.

### Author claim

The authors argue that conventional hallucination detection can say that an answer is unreliable without identifying where the failure originated.

### Our inference

This is almost exactly the conceptual move behind the original C1:

> “Do not only detect failure; locate the failure stage.”

Therefore, a generic stage-wise Medical RAG failure-analysis project should be considered **duplicate / very close**.

---

## Prior Work #3 — Benchmarking Retrieval-Augmented Generation for Medicine / MIRAGE / MedRAG

- **Title:** Benchmarking Retrieval-Augmented Generation for Medicine
- **Authors:** Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang
- **Year:** 2024
- **Venue:** Findings of ACL 2024
- **Status:** **Published**
- **DOI:** 10.18653/v1/2024.findings-acl.372
- **URL:** https://aclanthology.org/2024.findings-acl.372/
- **Primary / Secondary:** Primary research
- **Evidence strength:** High
- **Candidate threatened:** C1, C2

### What it does

MIRAGE contains 7,663 questions from five medical QA datasets.

MedRAG evaluates combinations of:

- medical corpora;
- retrievers;
- backbone LLMs.

The study examines how different components affect medical RAG performance and reports phenomena including “lost-in-the-middle”.

### Relevance

It does not perform the same fine-grained failure attribution as Kim et al. or StagRAG.

However, it establishes that medical RAG evaluation already includes systematic component variation.

### Implication

A proposal claiming:

> “We will compare retrieval models and LLMs to determine which component matters.”

is not novel by itself.

---

## Prior Work #4 — Do LLMs Know When Evidence is Insufficient?

- **Title:** Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in Retrieval-Augmented Generation
- **Authors:** Hantian Zhang, Wentai Wu
- **Year:** 2026
- **Venue:** Computers, Materials & Continua, 89(1)
- **Status:** **Published**
- **DOI:** 10.32604/cmc.2026.086343
- **URL:** https://doi.org/10.32604/cmc.2026.086343
- **Primary / Secondary:** Primary research
- **Evidence strength:** High
- **Candidate threatened:** C3

### What it does

The authors introduce a five-level evidence condition:

- L1: Full Support
- L2: Partial Support
- L3: Irrelevant Evidence
- L4: No Context
- L5: Conflicting Evidence

They evaluate seven LLMs from five model families.

They measure:

- accuracy;
- abstention rate;
- over-answer rate;
- wrong-answer rate;
- evidence sufficiency curves.

### Source fact

Under conflicting evidence, the evaluated models showed high over-answer rates, reported at 65–91%.

### Implication for C3

The proposition:

> “LLMs should recognize insufficient evidence and abstain”

is no longer an untested research direction.

The broad gap has already been formulated, benchmarked, and empirically demonstrated.

---

## Prior Work #5 — Evaluating Evidence Verification and Selective Abstention for Reliable Retrieval-Augmented Biomedical QA

- **Author:** Divya Dadi
- **Year:** 2026
- **Status:** SSRN preprint / working paper
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7391678
- **Primary / Secondary:** Primary research / preprint
- **Evidence strength:** Medium
- **Candidate threatened:** C3

### What it does

The study uses PubMedQA and a biomedical RAG pipeline containing:

- dense retrieval;
- hybrid retrieval;
- cross-encoder reranking;
- FLAN-T5 generation;
- NLI-based evidence verification.

It constructs insufficient-evidence cases by removing gold-associated passages and calibrates a reliability score for abstention.

### Source fact

The reported held-out test results include:

- abstention precision: 89.62%;
- abstention recall: 65.07%;
- abstention F1: 75.40%;
- coverage: 86.75%.

The author reports only a small change in answer accuracy after selective abstention.

### Implication

Even a medically/biomedically specialized version of:

> “use evidence verification to decide when to abstain”

already has direct precedent.

A new project would need to study a substantially different mechanism or experimental condition.

---

## Prior Work #6 — EHR-RAG

- **Title:** EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation
- **Authors:** Lang Cao, Qingyu Chen, Yue Guo
- **Year:** 2026
- **Status:** arXiv preprint
- **DOI:** 10.48550/arXiv.2601.21340
- **URL:** https://arxiv.org/abs/2601.21340
- **Primary / Secondary:** Primary research
- **Evidence strength:** High, but peer-review status should be treated separately from the technical evidence
- **Candidate threatened:** C4

### What it does

The paper explicitly studies long-horizon structured EHR data.

Its framework includes:

- Event- and Time-Aware Hybrid EHR Retrieval;
- Adaptive Iterative Retrieval;
- Dual-Path Evidence Retrieval and Reasoning.

The motivation directly concerns:

- long patient histories;
- context limits;
- temporal dependencies;
- incomplete evidence coverage.

### Source fact

The paper reports experiments across four long-horizon EHR prediction tasks and reports an average Macro-F1 improvement of 10.76% over its strongest LLM-based baselines.

### Implication

“Longitudinal EHR RAG” is already an active research area.

Therefore:

> “We will build a longitudinal patient RAG system.”

is not novel.

---

## Prior Work #7 — Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction

- **Authors:** Maya Kruse et al.
- **Year:** 2025
- **Venue:** Findings of EMNLP 2025
- **Status:** Published
- **URL:** https://aclanthology.org/2025.findings-emnlp.1128/
- **Primary / Secondary:** Primary research
- **Candidate threatened:** C4

### What it studies

The paper evaluates open-source LLMs, RAG variants, and CoT approaches on long clinical trajectories.

It focuses on:

- structured and unstructured EHR data;
- temporal coherence;
- longitudinal clinical summarization;
- diagnosis prediction.

### Implication

Temporal reasoning over longitudinal patient records is already empirically studied.

C4 therefore cannot claim novelty merely from adding “longitudinal” to Medical RAG.

---

## Prior Work #8 — Investigating RAG-based Approaches in Clinical Trial and Patient Matching

- **Authors:** Daniel León Tramontini, Shrestha Ghosh, Carsten Eickhoff
- **Year:** 2026
- **Venue:** Proceedings of the Fifth Machine Learning for Health Symposium, PMLR 297
- **Status:** Proceedings paper
- **URL:** https://proceedings.mlr.press/v297/leon-tramontini26a.html
- **Primary / Secondary:** Primary research
- **Candidate threatened:** C3, C4

### What it studies

The work explicitly examines:

- task complexity;
- retrieval from longitudinal records;
- abstention;
- chunking longitudinal patient records.

It reports noticeable effects of longitudinal chunking and abstention on performance.

### Implication

The combination:

> longitudinal patient records + RAG + abstention

is already being investigated.

---

## Prior Work #9 — IPF-RAG

- **Title:** IPF-RAG: Temporal-Aware Retrieval-Augmented Generation for Idiopathic Pulmonary Fibrosis Care
- **Year:** 2026
- **Status:** Accepted by Frontiers in Digital Health; formatted publication was indicated as forthcoming at the time of search
- **URL:** https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1883739/abstract
- **Primary / Secondary:** Primary research
- **Candidate threatened:** C4

### What it does

The system explicitly addresses:

- longitudinal disease progression;
- temporal retrieval;
- patient trajectories;
- clinical decision support.

The study uses patient trajectories and evaluates generated treatment plans with automated and clinical-expert assessments.

### Implication

Temporal-aware patient-specific RAG is not merely theoretical; it is being implemented and clinically evaluated in 2026 work.

---

## Prior Work #10 — RAG in Clinical Practice: A Cautionary Tale of AI “Truthfulness”

- **Authors:** HyoJe Jung et al.
- **Year:** 2026
- **Venue:** npj Health Systems
- **Status:** **Published**
- **Publication date:** 2026-07-03
- **URL:** https://www.nature.com/articles/s44401-026-00115-x
- **Primary / Secondary:** Primary research
- **Candidate threatened:** C2, C4

### What it studies

The authors repeatedly query a clinical RAG system while increasing dialogue history.

### Source fact

The reported hallucination rate increased from 5% without history to 40% after 10 prior exchanges.

### Important implication

This is especially important for any proposal involving:

> persistent conversational patient memory.

The failure mechanism can emerge from **history accumulation**, even without inventing a new retrieval architecture.

Therefore, persistent memory cannot be treated as a blank research space.

---

# E3 — Deep Comparative Extraction

## Master comparison

| Dimension | C1 Stage-wise failure | C2 Correct retrieval → wrong generation | C3 Evidence sufficiency / abstention | C4 Longitudinal patient RAG |
|---|---|---|---|---|
| Problem | RAG failure localization | Post-retrieval failure | Knowing when evidence is insufficient | Failure under longitudinal patient context |
| Closest prior | Kim 2025; StagRAG 2025 | Kim 2025 | Zhang & Wu 2026; Dadi 2026 | EHR-RAG 2026; Kruse 2025; IPF-RAG 2026 |
| Retrieval studied | Yes | Yes | Yes | Yes |
| Evidence selection | Yes | **Yes** | Yes | Yes |
| Generation failure | Yes | **Yes** | Indirectly | Yes |
| Evidence sufficiency | Partially | Potentially | **Explicitly** | Yes, via completeness |
| Abstention | Not central | Not central | **Explicitly** | Emerging |
| Longitudinal context | Limited | Limited | Generally not medical longitudinal | **Explicitly** |
| Error attribution | **Explicit** | Partially | Behavioral calibration | Emerging |
| Clinical data | Yes | Yes | Mostly biomedical/general RAG | **Yes** |
| Strongest overlap | **Very strong** | Strong | **Very strong** | Strong and rapidly increasing |

---

# E4 — Novelty Matrix

| Dimension | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Problem overlap | Strong | Strong | Strong | Strong |
| Research-question overlap | Strong | Partial–Strong | Strong | Partial–Strong |
| Failure mechanism overlap | Strong | **Strong** | Strong | Partial–Strong |
| Data overlap | Partial | Partial | Partial | **Strong** |
| Patient data overlap | Partial | Partial | Low–Partial | **Strong** |
| Medical domain overlap | Strong | Strong | Partial–Strong | Strong |
| Retrieval overlap | Strong | Strong | Strong | Strong |
| Evidence selection overlap | Strong | **Strong** | Strong | Strong |
| Evidence sufficiency overlap | Partial | Partial–Strong | **Very strong** | Partial–Strong |
| Reasoning overlap | Strong | **Strong** | Partial | Strong |
| Generation overlap | Strong | **Very strong** | Partial | Strong |
| Abstention overlap | Low | Low–Partial | **Very strong** | Partial–Strong |
| Memory overlap | Low | Low | Low | **Strong** |
| Longitudinal overlap | Low | Low | Low–Partial | **Very strong** |
| Failure attribution overlap | **Very strong** | Strong | Partial | Emerging |
| Evaluation overlap | Strong | Strong | **Strong** | Strong |
| Clinical evaluation overlap | Partial | Partial | Partial | Strong |
| Main contribution overlap | **Very strong** | Strong | **Very strong** | Strong |

---

# E5 — Overlap Analysis

## C1 — Stage-wise Medical RAG failure

### Problem overlap
**Strong**

### Method overlap
**Strong**

### Data overlap
**Partial**

### Setting overlap
**Strong**

### Evaluation overlap
**Strong**

### Contribution overlap
**Very strong**

### Overall overlap judgment

> **DUPLICATE / VERY CLOSE**

The combination of Kim et al. and StagRAG is decisive.

Kim et al. already provides expert stage-wise evaluation of retrieval, evidence selection, and generation.

StagRAG explicitly frames the task as stage-wise error diagnosis and attribution.

### Decision

> **STOP as originally formulated.**

Do not build:

> “A stage-wise framework for analyzing Medical RAG failures.”

That formulation has already been substantially addressed.

---

# E6 — C2: Correct Retrieval → Incorrect / Incomplete Generation

This candidate is more interesting.

The key observation from Kim et al. is:

> Even when relevant passages are available, the model may fail to select or incorporate them correctly.

This creates a narrower possible problem.

But the phrase:

> “correct retrieval but wrong generation”

is still too broad.

## Existing work already solves

Existing work has already demonstrated:

- relevant retrieval does not guarantee correct output;
- evidence selection can fail;
- irrelevant context can degrade factuality;
- generation can remain incorrect under RAG;
- stage-wise evaluation can identify these failures.

## Existing work partially solves

What is less clear from the current search is whether the field has fully separated:

1. retrieval relevance;
2. evidence sufficiency;
3. evidence selection;
4. reasoning transformation;
5. final generation.

Kim et al. gets close, but its evidence-selection stage focuses heavily on whether retrieved references are incorporated/cited.

That leaves a potentially interesting distinction:

> **The evidence is both relevant and sufficient, but the model transforms it incorrectly during reasoning/generation.**

This is materially different from:

> “retrieval succeeded.”

Because retrieval success alone does not prove evidence sufficiency.

### Candidate residual gap

**Candidate gap — requires further verification.**

A stronger formulation would require a controlled intervention:

- provide the same gold/sufficient evidence to the model;
- remove retrieval uncertainty;
- independently verify evidence sufficiency;
- then measure residual reasoning/generation errors.

The scientific question becomes:

> Once retrieval and evidence sufficiency are controlled, what mechanisms still cause medical RAG generation errors?

That is much stronger than “RAG sometimes gets the answer wrong.”

---

# E7 — C3: Evidence Sufficiency / Abstention

## Existing work already solves

The broad concept is already established.

Zhang & Wu (2026) explicitly construct a five-level evidence-sufficiency benchmark.

Dadi (2026) directly evaluates selective abstention for biomedical RAG.

Additional medical literature discusses abstention and uncertainty.

Therefore:

> “Can a medical RAG system abstain when evidence is insufficient?”

is **not novel**.

## What remains potentially interesting

A narrower question could investigate:

> whether evidence sufficiency can be distinguished from evidence relevance in a medical setting.

For example:

- relevant but incomplete evidence;
- individually correct pieces that are jointly insufficient;
- temporally conflicting patient evidence;
- evidence that supports one conclusion but omits a critical contraindication;
- multiple retrieved passages that are individually relevant but jointly contradictory.

This is more scientifically meaningful than simply adding an abstention module.

### Candidate status

> **C3 broad = STOP.**

> **C3 specialized mechanism = POSSIBLE PIVOT.**

---

# E8 — C4: Longitudinal / Patient-specific RAG

This is the most dynamic candidate.

## Existing work already solves

The literature now contains:

- long-horizon EHR RAG;
- temporal-aware retrieval;
- longitudinal clinical summarization;
- patient-specific retrieval;
- longitudinal patient matching;
- temporal RAG;
- patient-memory / conversational-history concerns;
- longitudinal clinical event verification.

Therefore:

> “Build a patient-specific longitudinal RAG system”

is not novel.

## What remains potentially unresolved

The stronger problem is not merely longitudinal retrieval.

It is:

> **How should a RAG system behave when temporally separated evidence conflicts, when retrieved evidence is relevant but insufficient, and when the patient's current state must be distinguished from historical states?**

This is a **failure-mechanism** question.

However, even this is becoming crowded.

MedEventGraph-RAG (2026) explicitly addresses:

- temporal mismatch;
- repeated documentation;
- conflicting findings;
- patient-specific event graphs;
- supporting and contradicting evidence;
- insufficient outcomes.

Therefore, even “temporal contradiction + insufficient evidence” cannot automatically be claimed as novel.

---

# E9 — Backward Citation Findings

## Principle

A recent paper is not necessarily the origin of its idea.

The audit therefore distinguishes:

- direct prior work;
- inherited evaluation ideas;
- inherited RAG methodology;
- genuinely new formulation.

| Prior Paper | Earlier Work / Lineage | What is inherited? | Novelty implication |
|---|---|---|---|
| Kim et al. 2025 | Lewis et al. 2020 RAG; RAG evaluation literature; medical RAG surveys/benchmarks | retrieval-augmented generation + evaluation decomposition | Stage-wise Medical RAG is an extension, not a new general RAG concept |
| StagRAG 2025 | RAG evaluation + hallucination/error diagnosis literature | component-level error attribution | Further weakens novelty of generic failure localization |
| MedRAG 2024 | RAG + biomedical QA benchmarks | systematic component benchmarking | Weakens “compare retriever/LLM/corpus” novelty |
| Zhang & Wu 2026 | abstention / selective prediction + RAG evaluation | evidence-aware abstention | Weakens generic evidence-sufficiency novelty |
| Dadi 2026 | retrieval confidence + evidence verification + biomedical QA | selective abstention | Weakens biomedical abstention novelty |
| EHR-RAG 2026 | EHR modeling + RAG + temporal retrieval | long-horizon patient evidence retrieval | Weakens generic longitudinal-RAG novelty |

## Important conclusion

Backward citation analysis shows that the candidate ideas are not emerging from a vacuum.

The recent Medical RAG papers combine older RAG, retrieval evaluation, uncertainty, and clinical NLP concepts.

Therefore:

> A contribution must be located at the level of a **specific unresolved mechanism**, not simply at the level of combining established components.

---

# E10 — Forward Citation Findings

Because several of the most threatening papers are from late 2025 or 2026, citation accumulation is still immature.

This creates an important limitation:

> Absence of forward citations does **not** mean absence of follow-up work.

Instead, later 2026 research was searched by concept and by direct paper/title references.

## Observed follow-up / adjacent 2026 evidence

### Rethinking RAG → medical RAG tooling and follow-up work

The Yale-BIDS medical-RAG repository now explicitly cites the Kim et al. work as its primary project reference and implements retrieval/query-reformulation experiments derived from that research line.

This demonstrates active continuation rather than an isolated paper.

### Longitudinal RAG → multiple 2026 directions

Search identified:

- EHR-RAG;
- IPF-RAG;
- MedEventGraph-RAG;
- LongMedBench;
- longitudinal patient matching;
- temporal clinical reasoning;
- conversational-history failure.

### Implication

C4 is moving rapidly.

A novelty claim based on a 2025 literature snapshot would now be outdated.

---

# E11 — Novelty Threat Map

| Candidate | Closest Prior Work | Overlap | Remaining Difference | Threat |
|---|---|---|---|---|
| C1 | Kim 2025; StagRAG 2025 | Very strong | Mostly implementation/evaluation variations | 🔴 High |
| C2 | Kim 2025; StagRAG 2025 | Strong | Potentially isolate post-sufficiency reasoning failure | 🟠 Medium |
| C3 | Zhang & Wu 2026; Dadi 2026 | Very strong | Medical-specific conflict/sufficiency mechanisms | 🔴 High |
| C4 | EHR-RAG 2026; IPF-RAG 2026; MedEventGraph-RAG 2026 | Strong | A narrowly controlled longitudinal failure mechanism | 🟠 Medium |

---

# E12 — Residual Gap Analysis

## C1

### Existing work already solves

Stage-wise decomposition of Medical RAG.

### Existing work partially solves

Fine-grained causal attribution across all possible RAG sub-stages.

### Existing work does not yet adequately address

No sufficiently strong evidence was found for a universally accepted causal taxonomy covering every Medical RAG failure mode across every clinical setting.

### Why this does NOT rescue C1

A missing universal taxonomy is not automatically a research gap.

The candidate would need to demonstrate:

- a clinically meaningful unresolved failure class;
- reproducible annotation;
- independent evaluation;
- better explanatory power than existing frameworks.

### Confidence

**High**

### Verdict

**STOP / PIVOT**

---

## C2

### Existing work already solves

Relevant retrieval does not guarantee correct generation.

### Existing work partially solves

Separating retrieval failure from evidence-selection failure from generation failure.

### Existing work does not yet adequately address

A tightly controlled experiment in which:

1. evidence relevance is fixed;
2. evidence sufficiency is independently established;
3. retrieval variance is removed;
4. the model receives identical evidence;
5. residual reasoning/generation errors are attributed.

### Why this matters

This would allow a stronger causal claim:

> the system failed despite sufficient evidence.

That is much more informative than:

> the RAG answer was wrong.

### Evidence

Kim et al. explicitly report low evidence-selection performance and generation degradation even under RAG.

Therefore the phenomenon is real.

The unresolved question is whether the residual mechanism can be isolated cleanly.

### Confidence

**Medium**

### Verdict

**NARROW**

---

## C3

### Existing work already solves

Evidence sufficiency and abstention as research problems.

### Existing work partially solves

Medical-specific evidence sufficiency under complex clinical contexts.

### Existing work does not yet adequately address

Potentially:

> evidence that is individually relevant but jointly insufficient or temporally contradictory in clinical patient contexts.

But MedEventGraph-RAG and other 2026 work already move directly toward this problem.

### Confidence

**Low–Medium**

### Verdict

**STOP broad formulation; PIVOT only if a stronger mechanism is identified.**

---

## C4

### Existing work already solves

Longitudinal patient-specific RAG.

### Existing work partially solves

Temporal reasoning and long-horizon retrieval.

### Existing work does not yet adequately address

Potentially:

- longitudinal failure attribution under controlled evidence conditions;
- interaction between temporal conflict and generation;
- distinguishing current-state evidence from stale historical evidence;
- clinically meaningful error propagation across multiple visits.

### Important warning

Several 2026 papers are already moving directly into these areas.

Therefore this is not an easy gap.

### Confidence

**Medium**

### Verdict

**NARROW / PIVOT**

---

# E13 — Test Whether the Difference Is Actually Novel

## Test 1 — Different dataset

### Verdict

Not enough.

A Vietnamese medical dataset, MIMIC-derived dataset, or specialty-specific dataset is not novelty by itself.

It must expose a mechanism that existing benchmarks cannot test.

---

## Test 2 — Different model

### Verdict

Reject as novelty.

Replacing GPT-4o with Llama, Qwen, Gemini, etc. is not a research contribution unless the model difference is itself the scientific variable.

---

## Test 3 — Different domain

### Verdict

Not enough.

“Healthcare” is already the domain.

A specialty such as oncology, nephrology, ophthalmology, or pulmonary fibrosis only becomes scientifically meaningful if it introduces a failure mode that is not adequately represented elsewhere.

---

## Test 4 — More modules

### Verdict

Reject as novelty.

Adding:

- reranker;
- verifier;
- agent;
- memory;
- query reformulation;

does not create novelty automatically.

---

## Test 5 — Different metric

### Verdict

Not enough.

A new metric is meaningful only if existing metrics fail to measure an important clinical property.

---

## Test 6 — Different research question

### Verdict

**Strongest candidate for real novelty.**

The most defensible path is therefore:

> control the earlier stages and ask a new causal question about what remains.

---

# E14 — Red-Team Novelty Test

# C1 Red-Team

## Current Novelty Claim

“We propose a stage-wise framework to determine where Medical RAG fails.”

## Strongest Prior Work Against It

Kim et al. 2025 + StagRAG 2025.

## Why the prior work may invalidate the claim

Both directly address stage-wise medical RAG failure analysis.

## Strongest argument against our novelty

> “This is essentially the same research problem, with a different taxonomy and perhaps a different dataset.”

## Reviewer verdict

> **Reject as duplicate.**

---

# C2 Red-Team

## Current Novelty Claim

“Medical RAG can fail even when retrieval succeeds.”

## Strongest Prior Work Against It

Kim et al. 2025.

## Why the prior work may invalidate the claim

The paper explicitly reports that even relevant retrieved passages may not be correctly selected or used, and response factuality/completeness can decline.

## Strongest argument for remaining distinction

A new study could remove retrieval uncertainty entirely and experimentally establish:

> sufficient evidence → same evidence → different reasoning/generation outcomes.

This would isolate a residual mechanism.

## Aggressive reviewer objection

> “You have simply renamed evidence-selection failure as generation failure.”

## Required answer

The project must operationally define:

- retrieval success;
- evidence sufficiency;
- evidence selection;
- reasoning transformation;
- generation.

If these are not separately measured, the novelty claim collapses.

## Reviewer verdict

> **Major narrowing required.**

---

# C3 Red-Team

## Current Novelty Claim

“We study when Medical RAG should abstain because evidence is insufficient.”

## Strongest Prior Work Against It

Zhang & Wu 2026 + Dadi 2026.

## Aggressive reviewer objection

> “Evidence sufficiency and selective abstention have already been benchmarked, including biomedical QA.”

## Reviewer verdict

> **Reject broad formulation.**

Potentially salvageable only through a clearly new medical mechanism.

---

# C4 Red-Team

## Current Novelty Claim

“We study failure in longitudinal patient-specific RAG.”

## Strongest Prior Work Against It

EHR-RAG 2026, IPF-RAG 2026, Kruse et al. 2025, MedEventGraph-RAG 2026.

## Aggressive reviewer objection

> “Longitudinal retrieval, temporal reasoning, patient-specific evidence, contradiction, and insufficient evidence are already active research topics.”

## Reviewer verdict

> **Major narrowing required.**

---

# E15 — Candidate Research Questions After Novelty Audit

Only candidates surviving the audit should receive a refined RQ.

## Candidate survivor: C2

### Research Question

> **When medical evidence is independently verified as relevant and sufficient, what residual reasoning or generation failure modes cause a RAG system to produce incorrect or incomplete clinical answers, and can those failures be reliably attributed under controlled evidence conditions?**

This is materially stronger than:

> “Why does RAG hallucinate?”

It specifies:

- system: Medical RAG;
- condition: evidence is relevant and sufficient;
- failure: incorrect/incomplete output;
- mechanism: residual reasoning/generation;
- outcome: measurable failure attribution.

### Required experimental isolation

The experiment must compare at least:

1. No external evidence
2. Retrieved evidence
3. Gold/sufficient evidence
4. Gold evidence + evidence-selection constraint

Without this structure, the causal claim is weak.

---

# E16 — Hypothesis

## H1

> **When relevant and sufficient medical evidence is held constant, a non-trivial proportion of remaining RAG errors will arise from evidence misinterpretation, incomplete evidence integration, or reasoning-to-generation transformation rather than retrieval failure.**

A second hypothesis can be tested only if the first is supported:

> **H2: Fine-grained attribution of residual errors identifies failure mechanisms that aggregate answer-level accuracy and retrieval metrics fail to distinguish.**

These are experimentally testable.

---

# E17 — Contribution

If C2 survives the next validation round, the contribution should be framed as:

### Primary contribution

**New empirical finding / failure attribution**

Not:

> “a better RAG architecture.”

### Possible secondary contribution

**New evaluation protocol**

Specifically, a controlled protocol separating:

> retrieval correctness → evidence sufficiency → evidence selection → reasoning/generation.

### What should NOT be claimed

Do not claim:

- a new LLM;
- a new embedding;
- a new vector database;
- a new agent;
- “RAG solves hallucination”;
- universal clinical safety.

---

# E18 — Feasibility Check

## Data availability

Potentially feasible using public biomedical/medical QA datasets.

However, the strongest design requires **evidence-level annotation**, not just question-answer labels.

### Minimum data structure

Each example should contain:

- query;
- candidate retrieved passages;
- gold/supporting evidence;
- sufficiency label;
- expected answer;
- generated answer;
- statement-level correctness;
- failure attribution.

## Compute

A controlled evaluation can be implemented without training a new foundation model.

Possible compute:

- local RTX 3050/4050 for embeddings, reranking, lightweight models;
- Colab for experiments;
- API-based LLM inference where permitted.

### Important

The bottleneck is **not GPU compute**.

The bottleneck is:

> annotation quality + experimental design.

## Model access

Existing LLMs are sufficient.

Do not train a large model from scratch.

## Experiment time

Potentially one semester **if the scope is tightly controlled**.

A full four-candidate investigation is not feasible.

## Baselines

Possible baseline structure:

- vanilla RAG;
- oracle-evidence condition;
- no-RAG condition;
- evidence-filtered condition.

## Risk

### High risk

The supposed residual failure may disappear once sufficient evidence is supplied.

That would invalidate the hypothesis.

This is scientifically acceptable.

### Another high risk

The failure may actually be evidence selection, not generation.

That is why evidence selection must be separately measured.

---

# E19 — Recommended Experimental Logic

The most important design principle is:

> **Do not ask the system to diagnose its own failure without intervention.**

Instead, manipulate one variable at a time.

## Factor A — Retrieval

- noisy retrieval;
- relevant retrieval;
- oracle retrieval.

## Factor B — Evidence sufficiency

- sufficient;
- partially sufficient;
- insufficient;
- conflicting.

## Factor C — Evidence presentation

- raw retrieved set;
- filtered evidence;
- oracle evidence.

## Factor D — Generation

Measure:

- factual correctness;
- completeness;
- unsupported claims;
- contradiction;
- evidence utilization.

## Desired attribution matrix

| Retrieval | Sufficiency | Output | Candidate failure |
|---|---|---|---|
| Fail | insufficient | wrong | retrieval/evidence failure |
| Pass | insufficient | wrong | sufficiency failure |
| Pass | sufficient | wrong | residual reasoning/generation failure |
| Pass | sufficient | incomplete | evidence integration/completeness failure |
| Pass | conflicting | confident answer | conflict handling failure |

This is much more scientifically informative than a single RAG accuracy number.

---

# E20 — What Would Falsify the Proposed Direction?

A strong project should define failure conditions before implementation.

The direction should be abandoned or substantially revised if:

1. Oracle evidence makes the errors disappear almost completely.
2. Most residual errors are actually caused by evidence selection.
3. Existing StagRAG/Kim-style attribution already explains the residual errors.
4. The annotation protocol cannot reliably distinguish evidence sufficiency from relevance.
5. Inter-annotator agreement is too low.
6. The effect disappears across different LLMs.
7. The sample size is too small to estimate the proposed failure rates.
8. The experiment becomes merely a model comparison.

---

# FINAL NOVELTY VERDICT

## Candidate Ranking

| Rank | Candidate | Novelty | Evidence | Feasibility | Verdict |
|---:|---|---|---|---|---|
| 1 | C2 — controlled residual failure after sufficient evidence | Medium | Medium | High | **NARROW** |
| 2 | C4 — specific longitudinal failure mechanism | Medium | Medium | Medium | **PIVOT/NARROW** |
| 3 | C3 — evidence sufficiency / abstention | Low in broad form | High | High | **STOP / PIVOT** |
| 4 | C1 — generic stage-wise failure analysis | Very low | Very high | High | **STOP** |

---

# Recommended Research Direction

## Primary direction

> **Controlled attribution of residual Medical RAG errors after retrieval and evidence sufficiency are held constant.**

The key scientific question is not:

> “Where does RAG fail?”

That has already been substantially studied.

The stronger question is:

> **“After retrieval and evidence sufficiency are controlled, what failure mechanisms remain in medical RAG reasoning and generation, and can they be measured reproducibly?”**

This is a **candidate research direction**, not yet a guaranteed research gap.

It requires another verification round against the full papers and their latest 2026 follow-up literature before BUILD.

---

# Secondary direction

A secondary option is:

> **Longitudinal Medical RAG under controlled temporal conflict and evidence sufficiency.**

But this should remain secondary because 2026 literature is moving rapidly into:

- temporal RAG;
- longitudinal EHR;
- patient memory;
- conflicting evidence;
- clinical event verification.

---

# What Existing Work Already Solves

The literature already establishes that:

1. Medical RAG can fail at retrieval.
2. Medical RAG can retrieve irrelevant evidence.
3. Relevant evidence may not be selected or used correctly.
4. RAG may degrade factuality/completeness.
5. Stage-wise error attribution is already proposed.
6. Evidence sufficiency is already benchmarked.
7. Selective abstention has already been studied in biomedical RAG.
8. Long-horizon EHR RAG is already an active research direction.
9. Temporal patient reasoning is already being studied.
10. Longitudinal evidence conflict and event verification are already emerging as explicit 2026 research topics.

---

# What Remains

The strongest residual possibility is:

> **isolating residual reasoning/generation errors under experimentally controlled sufficient evidence.**

But this remains a:

> **Candidate gap — requires further verification.**

It should not yet be advertised as “novel”.

---

# Why Our Work Could Be Different

Only if the experiment enforces:

1. retrieval is controlled;
2. evidence relevance is independently verified;
3. evidence sufficiency is independently verified;
4. evidence is held constant;
5. output errors are decomposed at statement level;
6. failure attribution is reproducible;
7. the mechanism is compared across models/settings.

The scientific contribution would then be the **empirical isolation of a failure mechanism**, not another RAG architecture.

---

# BUILD GATE

## Problem:

Medical RAG can produce incorrect or incomplete clinical answers even when relevant external evidence is available. Existing work has already demonstrated stage-wise failures, evidence-selection failures, and general generation degradation. The remaining candidate problem is to determine whether a distinct residual reasoning/generation failure exists **after evidence relevance and sufficiency are controlled**.

## Research Question:

> When medical evidence is independently verified as relevant and sufficient, what residual reasoning or generation failure modes cause a RAG system to produce incorrect or incomplete clinical answers, and can those failures be reliably attributed under controlled evidence conditions?

## Hypothesis:

> H1: Under controlled conditions with relevant and sufficient evidence held constant, a non-trivial proportion of Medical RAG errors will persist and can be attributed to evidence misinterpretation, incomplete evidence integration, or reasoning-to-generation transformation rather than retrieval failure.

## Closest Prior Work:

1. Kim et al. (2025) — *Rethinking Retrieval-Augmented Generation for Medicine*
2. Wang et al. (2025) — *StagRAG*
3. Xiong et al. (2024) — *Benchmarking Retrieval-Augmented Generation for Medicine*
4. Zhang & Wu (2026) — *Do LLMs Know When Evidence is Insufficient?*
5. Dadi (2026) — *Evaluating Evidence Verification and Selective Abstention*
6. Cao et al. (2026) — *EHR-RAG*
7. Kruse et al. (2025) — longitudinal clinical reasoning/RAG
8. Jung et al. (2026) — conversational-history failure in clinical RAG
9. Lin et al. (2026) — *MedEventGraph-RAG*
10. Zheng et al. (2026) — *IPF-RAG*

## What Existing Work Already Solves:

- generic stage-wise Medical RAG diagnosis;
- retriever/reranker/generator attribution;
- broad evidence-sufficiency evaluation;
- biomedical selective abstention;
- long-horizon EHR retrieval;
- temporal clinical reasoning;
- several longitudinal patient-specific RAG mechanisms.

## What Remains:

Potentially the controlled isolation of **post-sufficiency residual reasoning/generation failure**, with retrieval and evidence sufficiency experimentally held constant.

## Why Our Work Is Different:

The distinction is experimental, not architectural:

> retrieval success → evidence sufficiency → fixed evidence → residual reasoning/generation.

If the study cannot enforce this separation, the novelty claim fails.

## Contribution:

Potentially:

- new empirical finding;
- controlled failure-attribution protocol;
- evidence-level evaluation framework.

Not a claim of a novel RAG architecture.

## Dataset:

Prefer a public medical QA benchmark with:

- expert answers;
- retrievable biomedical evidence;
- sufficient supporting passages;
- possibility of constructing controlled evidence conditions.

Candidate datasets should be selected only after verifying licensing, annotation quality, and evidence provenance.

## Baseline:

- no-RAG LLM;
- vanilla RAG;
- oracle-evidence RAG;
- evidence-filtered RAG.

## Proposed Method:

Controlled evidence intervention rather than a new complex architecture.

The core manipulation is:

> hold evidence constant while measuring residual model error.

## Evaluation:

### Retrieval

- Recall@k
- Precision@k
- Coverage

### Evidence

- relevance
- sufficiency
- completeness
- contradiction

### Generation

- factuality
- completeness
- unsupported-claim rate
- contradiction rate
- evidence utilization

### Attribution

- retrieval failure
- insufficiency
- evidence-selection failure
- reasoning/integration failure
- generation failure

## Ablation:

At minimum:

1. no retrieval;
2. noisy retrieval;
3. relevant retrieval;
4. sufficient oracle evidence;
5. sufficient evidence + filtering;
6. conflicting evidence.

## Expected Failure Cases:

- evidence is relevant but incomplete;
- evidence is sufficient but ignored;
- evidence is sufficient but misinterpreted;
- multiple evidence statements are individually correct but jointly misunderstood;
- model introduces unsupported claims;
- model produces incomplete answer despite sufficient evidence;
- model contradicts retrieved evidence.

## Compute:

Feasible without foundation-model training.

Primary costs are:

- LLM inference;
- retrieval infrastructure;
- annotation;
- evaluation.

GPU demand should be manageable on RTX 3050/4050 or Colab if local/open models are used.

## Semester Feasibility:

**Potentially feasible**, but only with one candidate and a tightly controlled dataset.

A four-candidate project is not feasible within one semester.

## Novelty Confidence:

**60/100 for the narrowed direction.**

This does **not** mean “60% novel”.

It means:

> The distinction is plausible and scientifically meaningful, but the existing 2025–2026 literature is close enough that a full-paper verification and pilot experiment are required before claiming novelty.

## FINAL DECISION

> **NARROW — DO NOT BUILD YET**

### Current status

The original candidates have been successfully filtered.
C1 and broad C3 are rejected; C4 remains crowded.
C2 is the strongest surviving candidate.

However, C2 has NOT yet passed final novelty verification.

### Required next step

> ROUND 2 — FULL-PAPER AND CITATION VERIFICATION

The candidate must not be promoted to a final Research Question or Build Gate until:
1. the closest prior papers are manually verified in full;
2. backward citation tracing is completed;
3. forward citation tracing is completed;
4. evidence sufficiency, evidence selection, reasoning, and generation are operationally separated;
5. the residual contribution survives a second red-team review.

### Current confidence

Novelty confidence: 60/100

Status:
> Candidate — promising but unverified
---

# Audit Limitations

This document is a research due-diligence snapshot, not a proof of global novelty.

Important limitations:

1. Citation accumulation for late-2025/2026 papers is still incomplete.
2. Some 2026 papers are preprints rather than peer-reviewed publications.
3. Search engines do not guarantee exhaustive retrieval of every paper.
4. “No citing paper found” must not be interpreted as “no follow-up exists”.
5. Novelty must ultimately be judged against the exact experimental formulation, not keywords.
6. The final project should not be frozen until the closest papers' full methods, tables, references, and supplementary material have been manually checked.

---

# Evidence Classification Rules

Every future claim should be tagged internally as:

- **Source fact** — directly observed from the paper.
- **Author claim** — explicitly claimed by the authors.
- **Our inference** — conclusion drawn by this audit.
- **Candidate gap — requires further verification** — plausible but insufficiently established.

Never convert:

> “Paper X did not evaluate Y”

directly into:

> “Y is a research gap.”

---

# Core Due-Diligence Principle

The burden of proof is asymmetric.

We do **not** need to prove:

> “Nobody has ever done this.”

We need to establish:

> **“After accounting for the strongest relevant prior work, the proposed research question remains scientifically meaningful, experimentally distinguishable, and feasible.”**

If that cannot be demonstrated:

> **STOP.**

That is a successful research decision.
