# CAF-RAG Full-Paper Verification Report — Novelty Verification Round 2

**Audit date:** 15 October 2026  
**Candidate under review:** C2 — residual errors under a fixed, independently verified sufficient medical evidence set  
**Verdict for CAF-RAG:** **B. PARTIAL OVERLAP; C2 MAY SURVIVE**  
**Overall C2 status:** **Still provisional; novelty is not cleared globally.**

---

## 1. Executive verdict

CAF-RAG is a close competitor to broad Medical RAG failure analysis and to studies asking when retrieval helps or harms. It directly compares retrieval and passage-selection interventions, analyzes question-level outcomes, and proposes a three-mode failure taxonomy. Therefore, it strengthens the decision to stop C1 in its generic “where does Medical RAG fail?” form.

However, in the full-text content available from the publisher's indexed article, **CAF-RAG does not establish the central experimental condition proposed by C2**: an identical evidence bundle that has been independently verified to be jointly sufficient, followed by measurement of errors specifically among cases that pass that sufficiency gate. It varies how passages are selected and measures final answers across those conditions. Relevance/clinical alignment and evidence sufficiency are not treated as interchangeable in this audit.

This yields a **partial-overlap verdict, not a novelty approval**. C2 may continue to the remaining bounded competitor checks and a small feasibility pilot. Kim et al. remains a direct threat, StagRAG remains an inaccessible full-text risk, and Zhang & Wu needs a targeted methods comparison.

## 2. Confirmed paper identity and source access

**Title:** *When does retrieval help medical question answering? A question-type sensitivity analysis and failure taxonomy for text-based (image-free) medical RAG, with an interpretable clinical filter (CAF-RAG)*  
**Authors shown in the article contribution statement:** Youssef Azami, Hamza Boulibya, Mohamed Adnane Mahraz, Mohammed Oulhadj, Hamid Tairi, Ilyasse Aboussaleh, Khalid El Fazazy, and Jamal Riffi.  
**Journal:** *Informatics in Medicine Unlocked*, Volume 65, Article 101810, September 2026.  
**DOI:** [10.1016/j.imu.2026.101810](https://doi.org/10.1016/j.imu.2026.101810)  
**Publisher page:** https://www.sciencedirect.com/science/article/pii/S2352914826000808

**Access note:** The publisher's indexed result exposed substantial article text, including the abstract, sections on datasets, knowledge base, retrieval, reranking, CAF filtering, experimental conditions, failure taxonomy, conclusion, and data availability. A direct page-open request in this session returned a cache-miss error, and I did not obtain a separately downloaded publisher PDF. Therefore, this report is based on the full-text content visible in the official publisher-indexed result, not on a locally downloaded PDF. Details unavailable in that material are marked unresolved rather than guessed.

## 3. Summary of CAF-RAG's actual method

CAF-RAG studies when **text-only medical retrieval helps, has no meaningful benefit, or harms answer accuracy**. Its questions come from SLAKE and VQA-RAD, with external validation on PubMedQA. The combined evaluation contains 1,512 questions, divided into SLAKE-CLOSED (416), SLAKE-OPEN (645), VQA-RAD-CLOSED (251), and VQA-RAD-OPEN (200). The images are intentionally not passed to the model; this is an image-free text QA study, not a multimodal RAG evaluation.

The reported knowledge base has 2,851 passages: 2,588 SLAKE-derived knowledge-graph triples and 263 PubMed abstracts. Six conditions are compared: No-RAG, dense RAG, hybrid BM25+dense retrieval, a general cross-encoder reranker, a biomedical cross-encoder reranker, and CAF-RAG (an interpretable clinical filter). The dense retrieval depth is set to k=3 for closed questions and k=7 for open questions. The filter scores modality/organ and related clinical-semantic signals, removes passages at or below its threshold, and has a minimum-context safeguard.

The methodology says the question, prompt template and model parameters are held fixed across retrieval conditions, while the supplied context changes. A dated GPT-4o-mini snapshot at temperature 0 is described for the primary inference setup; the article also reports a BioMistral-7B replication/comparison in its summary and results. Those statements appear to describe primary and replication experiments, but the exact split between them should be followed carefully when reproducing results.

Reported findings include: retrieval did not significantly outperform parametric knowledge on the tested groups; on PubMedQA, the paper reports a 12-percentage-point degradation (p<0.0001); on SLAKE-CLOSED, CAF-RAG improved over unfiltered RAG by 0.034 (p=0.027) but did not significantly exceed No-RAG. The paper also reports a lexical substring-matching scoring artifact and a replacement scorer validated against 197 blinded human gradings (91.9% agreement, κ=0.79). These results support its diagnostic contribution about retrieval effects and evaluation pitfalls; they do not, by themselves, establish the post-sufficiency residual-error condition required by C2.

## 4. Five-criterion verification table

| Criterion | Verdict | Evidence from article | Supporting location | Implication for C2 |
|---|---|---|---|---|
| 1. Oracle/gold evidence or equivalent controlled evidence intervention | **PARTIAL** | The study performs controlled comparisons among retrieval/selection conditions, including reranking and CAF filtering. No oracle/gold evidence condition that removes retrieval uncertainty was identified in the inspected methods. | §3.3–3.7; six experimental conditions; data availability | It intervenes on passage selection but does not demonstrate the specific gold/sufficient-evidence intervention proposed by C2. |
| 2. Exact evidence bundle held constant | **PARTIAL** | Questions, prompt, and generation settings are controlled across conditions, but the context supplied to the generator is explicitly the variable that changes. The article summary also says the two models are compared with byte-identical contexts; this does not mean contexts are held constant across all retrieval conditions. | Abstract; §3.7, experimental conditions and evaluation protocol | Not a controlled comparison of downstream generation over the same validated evidence bundle across the relevant conditions. |
| 3. Independent sufficiency verification | **NOT ESTABLISHED** | The filter uses clinical-semantic signals such as modality and organ alignment; answer scoring is validated against blinded human gradings. The inspected method does not define or independently verify whether the complete evidence bundle is jointly sufficient to answer each question. | §3.6–3.7; §4.5 as summarized in article text | This is the most important remaining distinction from C2. Clinical relevance, answer-scoring validation, and sufficiency are different properties. |
| 4. Evidence selection separated from reasoning/generation | **PARTIAL** | Selection methods are compared directly, but the selected context differs by condition. No condition was identified in which selection is shown to pass a prespecified independent sufficiency gate and then downstream generation is evaluated separately on that same evidence. | §3.3–3.7; §5 failure taxonomy | It studies selection's effect on answer accuracy; it does not isolate the residual error conditional on selection already passing. |
| 5. Residual errors measured after all upstream gates pass | **NOT ESTABLISHED** | Outcomes include final accuracy, question-type effects, answer-distribution shifts, and a retrieval failure taxonomy. No quantitative residual-error rate is reported for a subset where independent sufficiency has passed and the exact evidence bundle is fixed. | §4 results as summarized in the article text; §5; §7 | C2's narrow experimental claim is not directly reproduced by CAF-RAG, based on the inspected materials. |

**Important interpretation rule:** “NOT ESTABLISHED” means the inspected article material does not demonstrate the required condition. It does not prove that no related experiment exists anywhere else.

## 5. Comparison with C2

| Dimension | CAF-RAG | C2's current proposal | Overlap | Residual distinction |
|---|---|---|---|---|
| Research question | When does text-based retrieval help, do nothing, or harm medical QA, and can clinical filtering mitigate noise? | What answer errors remain with fixed, independently verified sufficient evidence, and does a grounding constraint change them? | PARTIAL | Different primary scientific question. |
| Dataset/task | SLAKE, VQA-RAD, external PubMedQA; image-free text condition | Not locked yet; must support per-question supporting evidence and sufficiency labels | PARTIAL | C2's dataset must be chosen for controlled evidence annotation, not simply reused because it is medical QA. |
| Retrieval control | Six conditions vary retrieval/selection approach | Hold evidence bundle fixed for the central comparison | PARTIAL | C2 controls a factor CAF-RAG intentionally varies. |
| Evidence relevance | Clinical-semantic filtering based on modality/organ signals | Independently judge evidence relevance before sufficiency assessment | PARTIAL | C2 needs operational labels, not just a filter score. |
| Evidence sufficiency | Joint sufficiency of full context not independently established in inspected methods | Independent sufficiency gate for each evidence bundle | LOW / NOT ESTABLISHED in CAF-RAG | This is the main potential distinction, provided it is operationalized reproducibly. |
| Fixed evidence | Context differs across retrieval conditions; cross-model byte-identical contexts are reported | Exact question–evidence pair fixed across prompt/intervention comparison | PARTIAL | C2 can isolate post-evidence effects if it really freezes the bundle. |
| Evidence selection | Selection methods are directly compared | Selection is eliminated or separately verified before evaluating downstream errors | PARTIAL | C2 must prevent selection errors from being renamed “generation errors.” |
| Reasoning/generation analysis | Aggregate final answer outcomes and behavioral shifts; retrieval-effect taxonomy | Claim-level errors conditioned on passed sufficiency, with controlled prompt intervention | PARTIAL | Needs careful protocol and annotation to prove more than output-level error inspection. |
| Controlled intervention | Changes retrieval/selection conditions, not an independently verified sufficient bundle | Same sufficient evidence; compare context-only versus explicit grounding constraint | PARTIAL | The proposed intervention differs, but its novelty against other papers is not yet cleared. |
| Main contribution | Empirical characterization of retrieval benefit/harm and an interpretable clinical filter | Controlled empirical attribution of residual errors after evidence sufficiency | PARTIAL | C2 must demonstrate a causal/measurement benefit rather than merely adopt another error taxonomy. |

## 6. Strongest adversarial argument against C2

A skeptical reviewer could say:

> “CAF-RAG already performs a controlled analysis of passage-selection methods and medical RAG failures. C2 merely changes the prompt and calls remaining answer errors ‘reasoning/generation failures.’ Without a reliable independent sufficiency label and evidence-use annotation, C2 cannot distinguish a genuine downstream reasoning failure from evidence omission, incompleteness, or a question whose answer is absent from the retrieved corpus.”

This objection is valid unless C2 specifies and tests all of the following:

1. Evidence relevance and joint sufficiency are labeled separately.
2. Sufficiency is judged independently from the model's answer.
3. Exact passages, their order, prompt context, model snapshot, and decoding settings are fixed across the relevant comparison.
4. Any learned evidence selection step is removed or shown to pass a separate criterion.
5. Output errors are evaluated at claim level, distinguishing unsupported claims, contradictions, omissions/incompleteness, and supported correct statements.

The fact that CAF-RAG has not established the exact C2 condition does not automatically make C2 novel; the remaining accessible competitors must still be checked.

## 7. Remaining distinction, if any

**Potential distinction supported against CAF-RAG:** a post-sufficiency experiment that fixes an independently validated evidence bundle and compares (a) a context-only prompt against (b) an explicit evidence-grounding constraint, measuring claim-level support and completeness. CAF-RAG primarily manipulates which context is retrieved/retained and studies retrieval effects, rather than reporting error rates conditional on an independently verified sufficiency gate.

This is a potential experimental distinction, not yet a verified literature gap. Kim et al. and Zhang & Wu remain necessary direct checks, while StagRAG remains `DIRECT OVERLAP UNRESOLVED` because its full text is inaccessible.

## 8. Limitations and unresolved items

- The publisher page could not be opened directly in the current browsing session; the audit used substantial full-text material surfaced by the publisher's indexed result. A separately downloaded PDF and its pagination were not available.
- The inspected material did not establish independent joint evidence-sufficiency labels or residual-error measurement after all upstream gates passed.
- The exact relationship between the primary GPT-4o-mini runs and the reported BioMistral-7B replication should be followed carefully before attempting reproduction.
- This report decides CAF-RAG's overlap with C2 only; it does not resolve overlap with Kim et al., StagRAG, or all of the literature.

## 9. Final CAF-RAG verdict

**B. PARTIAL OVERLAP; C2 MAY SURVIVE.**

CAF-RAG strongly overlaps with broad retrieval-failure analysis, passage-selection intervention, and answer-level evaluation. It does not, in the inspected methods/results, demonstrate the specific combination of fixed evidence + independently verified joint sufficiency + residual-error measurement conditional on passing that gate. Do not claim C2 is novel; retain it as provisional.

## 10. Explicit next action

**Perform a targeted methods audit of Zhang & Wu (2026)**, limited to how the five evidence conditions are constructed, whether “full support” means independently verified joint sufficiency, what is held fixed, and whether downstream errors are measured after an explicit sufficiency gate. In parallel, the project may start a 20–30-item dataset feasibility pilot; do not wait for StagRAG.

## 11. Sources and exact supporting locations

1. Azami et al., *When does retrieval help medical question answering? A question-type sensitivity analysis and failure taxonomy for text-based (image-free) medical RAG, with an interpretable clinical filter (CAF-RAG)*, *Informatics in Medicine Unlocked* 65 (September 2026), Article 101810. DOI: [10.1016/j.imu.2026.101810](https://doi.org/10.1016/j.imu.2026.101810). Publisher page: https://www.sciencedirect.com/science/article/pii/S2352914826000808. Relevant locations: §1 (research questions/contributions), §3.1 (datasets), §3.2 (knowledge base), §3.3–3.7 (retrieval, selection, conditions and prompts), §4 (results and scoring audit), §5 (failure taxonomy), §6 (limitations), §7 (conclusion/future work), and Data availability.
