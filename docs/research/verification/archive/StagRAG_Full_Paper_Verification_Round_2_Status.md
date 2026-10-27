# StagRAG Full-Paper Verification — Round 2 Status

**Audit date:** 10 October 2026  
**Status:** **NOT CLEARED — full-paper evidence unavailable in this audit session**  
**Purpose:** Determine whether StagRAG already implements the controlled experiment proposed for candidate C2.

## 1. Source-access log

Sources checked:

1. [Penn State publication record](https://pure.psu.edu/en/publications/stagrag-explaining-where-rag-systems-go-wrong-in-medical-reasonin/) — bibliographic record and abstract available.
2. [IEEE Xplore document record](https://ieeexplore.ieee.org/document/11402407) — page opens only to a JavaScript/robot-verification message in this session; the paper body was not retrievable.
3. [ResearchGate publication record](https://www.researchgate.net/publication/401690261_StagRAG_Explaining_Where_RAG_Systems_Go_Wrong_in_Medical_Reasoning) — explicitly states that no full text is available on the page and offers a request-full-text action.
4. Author code repository linked from the Penn State record: [YCWang0709/StagRAG](https://github.com/YCWang0709/StagRAG). Direct retrieval of the repository contents was unsuccessful in this session.

**Interpretation:** this is not a completed full-paper verification. It is a documented access attempt plus an abstract-level threat assessment. Failure to retrieve the paper is not evidence that the paper lacks any method or experiment.

## 2. Confirmed bibliographic facts

- **Title:** *StagRAG: Explaining Where RAG Systems Go Wrong in Medical Reasoning*.
- **Authors:** Yucheng Wang, Md Almahfuz Chowdhury, Mahmuda Akter Sristy, Md Faisal Kabir.
- **Venue:** 2025 IEEE International Conference on Big Data (BigData 2025), pages 8030–8039.
- **DOI:** [10.1109/BigData66926.2025.11402407](https://doi.org/10.1109/BigData66926.2025.11402407).

## 3. What the abstract establishes

The published abstract states that StagRAG:

- proposes a **stage-wise RAG diagnostic framework**;
- attributes errors to the **retriever (R), reranker (Re), or generator (G)**;
- uses **oracle-guided interventions** and **auxiliary inference probes**;
- reports failures associated with insufficient retrieval coverage and generator instability.

These details are sufficient to treat StagRAG as a direct threat to a generic proposal to localize Medical RAG failures by pipeline stage.

They are **not** sufficient to conclude that it tests residual generation errors after the exact same evidence has been independently verified as sufficient and selection uncertainty has been removed.

## 4. Five-point verification matrix

| Criterion | Status | Evidence available | What must be checked in the paper/code |
|---|---|---|---|
| 1. Oracle evidence | **Partially confirmed** | Abstract explicitly says “oracle-guided interventions.” | What the oracle returns; how gold evidence is defined; which stage is replaced; whether oracle means gold supporting passages or merely corrected retrieval/reranking. |
| 2. Evidence held fixed | **Unverified** | Abstract does not specify identical question–evidence pairs across the relevant conditions. | Whether the exact evidence passages, order, prompt, decoding settings, and question are held constant when comparing generator outcomes. |
| 3. Independent sufficiency verification | **Unverified** | Abstract does not give an operational definition or validation protocol for sufficient evidence. | Whether the whole evidence set is judged sufficient independently of the evaluated answer; who labels it; whether labels are at passage or set level; reliability/quality controls. |
| 4. Selection separated from generation | **Unverified** | Stage attribution to retriever/reranker/generator is confirmed, but stage labels alone do not establish a controlled selection-free generator test. | Whether selection is bypassed or manually/oracle-corrected; whether the generator receives a validated fixed evidence set without a learned selection stage changing it. |
| 5. Residual reasoning failure after selection passes | **Unverified** | Abstract reports generator instability, but no controlled-subset definition or residual error rate is available. | Quantitative outcomes on cases where evidence sufficiency and evidence selection have explicitly passed; whether omission, misinterpretation, integration, and unsupported generation are distinguished. |

**Status rule:** “Unverified” means evidence currently accessible did not settle the question. It does not mean the paper did not do it.

## 5. Novelty implications

### C1 — generic stage-wise failure localization

**Verdict: STOP.** The abstract is already enough to establish substantial conceptual overlap with a generic stage-wise Medical RAG diagnostic framework.

### C2 — residual failure with sufficient, fixed evidence

**Verdict: NOT CLEARED; do not BUILD yet.** A narrower distinction is still logically possible, but it has not been demonstrated. The burden is on the proposed project to show that its experimental control differs materially from StagRAG and other closely related work.

Do not claim any of the following until full-paper verification is complete:

- that StagRAG only repairs retrieval coverage;
- that StagRAG does not hold evidence fixed;
- that StagRAG does not assess sufficiency;
- that StagRAG does not measure generator failures after an oracle intervention.

None of those negative claims is supported by the available abstract-level evidence.

## 6. Required next evidence

**Preferred:** upload the paper PDF (10 pages, pp. 8030–8039), including any appendix/supplement if one exists. Then extract exact page/table/figure/method references for all five criteria.

**Alternative:** provide access to the public GitHub repository contents or upload a repository ZIP. The repository may clarify implemented interventions and experiment scripts, but source code alone may not reveal every paper-level definition, result, or limitation.

Once the full paper is available, record for each criterion:

1. exact page/section and table/figure/equation;
2. verbatim short evidence or precise paraphrase;
3. whether the point is an author claim, implemented procedure, or reported result;
4. whether it directly overlaps C2;
5. resulting novelty decision: STOP, NARROW, or survive to the next audit.

## 7. Sources

- Penn State publication record and abstract: https://pure.psu.edu/en/publications/stagrag-explaining-where-rag-systems-go-wrong-in-medical-reasonin/
- IEEE Xplore: https://ieeexplore.ieee.org/document/11402407
- DOI: https://doi.org/10.1109/BigData66926.2025.11402407
- ResearchGate record (currently states no full text available on the page): https://www.researchgate.net/publication/401690261_StagRAG_Explaining_Where_RAG_Systems_Go_Wrong_in_Medical_Reasoning
- Repository linked by the publication record: https://github.com/YCWang0709/StagRAG
