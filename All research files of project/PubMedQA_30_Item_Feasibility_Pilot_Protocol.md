# PubMedQA 30-Item Feasibility Pilot — Protocol v0.1

**Project phase:** Round 2 → initial dataset feasibility (parallel preparation for Research Design)  
**Status:** Protocol ready; pilot data have not yet been annotated or run through an LLM.  
**Purpose:** Test whether an accessible medical QA dataset can support a controlled experiment on errors remaining after evidence is independently judged sufficient. This pilot does NOT test the research hypothesis and does NOT prove novelty.

## 1. Why start with PubMedQA PQA-L?

The official PubMedQA project describes 1,000 expert-labelled, 61.2k unlabeled, and 211.3k artificially generated questions. PQA-L supplies a question, a corresponding abstract context without the conclusion, a long answer and a gold `yes/no/maybe` decision. The task requires answering biomedical research questions from the provided abstract. This makes PQA-L a practical first feasibility candidate, although its context sufficiency must be audited rather than assumed.

Primary sources:
- Official dataset page: https://pubmedqa.github.io/
- Official repository: https://github.com/pubmedqa/pubmedqa
- Original paper: Jin et al. (2019), ACL Anthology: https://aclanthology.org/D19-1259/

Use the expert-labelled PQA-L subset, NOT the artificial PQA-A subset. Keep the long answer and final decision hidden during the first evidence-sufficiency annotation.

## 2. Pilot scope

- Initial sample: 30 items, ideally 10 per gold class (`yes`, `no`, `maybe`) to ensure class coverage.
- Sampling seed: 42.
- Prefer excluding IDs in the official `test_ground_truth.json` so these pilot items do not contaminate a later reserved test set. The helper script attempts to download/exclude these IDs; if it cannot access that file, it must print a warning.
- This is a feasibility sample, not a statistically representative benchmark. Do not report its answer accuracy as a final performance estimate.

## 3. Blinding rule

**First annotate evidence sufficiency without seeing model outputs, `final_decision`, or `long_answer`.** The helper script generates two separate outputs:

1. `pubmedqa_pilot_30_blinded.csv`: question, evidence context, and blank annotation fields. It omits PubMed ID and gold answers to reduce lookup/label leakage.
2. `pubmedqa_pilot_30_key.json`: hidden mapping from pilot ID to PubMed ID and gold fields. Do not open it until the sufficiency annotation sheet has been saved.

The context contains the dataset's objective/methods/results passages, when available. Do not insert the long answer/conclusion into the supplied evidence context. Preserve the exact context provided to each model condition.

## 4. Evidence-sufficiency annotation rubric

Assign exactly one initial label per item:

- `SUFFICIENT`: given only the question and supplied context, a careful reader can support one answer (`yes`, `no`, or `maybe`) from the context without needing missing external facts.
- `PARTIAL`: some relevant information is present, but one or more necessary facts/relationships are missing; an answer requires material inference beyond the context.
- `INSUFFICIENT`: essential information needed to answer is absent.
- `CONFLICTING`: supplied evidence materially supports incompatible conclusions or contains unresolved contradiction.
- `AMBIGUOUS_OR_INVALID`: question/context is malformed, underspecified, or cannot be judged reliably.

Record a short note naming the supporting or missing information. Do not infer `SUFFICIENT` merely because the dataset contains a gold label or long answer. A gold label can be correct while the visible evidence is incomplete.

The initial pilot may be annotated by one student to assess workload, but the final research protocol should seek a second independent annotator for the evidence-sufficiency labels. Disagreements should be adjudicated; uncertain cases should not be forced into the sufficient subset.

## 5. Proposed model comparison after the annotation is locked

Only items independently labelled `SUFFICIENT` enter the primary post-sufficiency analysis. For each item, preserve the exact same question and evidence bundle in both conditions:

**Condition A — Context-only baseline**
- Ask for a `yes/no/maybe` answer and a concise explanation based on the supplied context.
- Do not provide the gold answer or long answer.

**Condition B — Explicit evidence-grounding constraint**
- Require the model to use only the supplied context.
- Ask it to cite the context block(s) supporting each substantive claim (use the labels already present in the context).
- Require it to say `insufficient evidence` rather than invent support if the context cannot justify an answer.
- Keep model version and decoding settings the same as Condition A.

Condition B is a hypothesis-generating intervention, not assumed to be novel. If used in the final experiment, prompt versions must be frozen and reported verbatim.

## 6. Outcome coding

After the blinded sufficiency decisions are frozen, unblind the gold fields and evaluate outputs with a pre-defined rubric:

1. `decision_correct`: predicted `yes/no/maybe` matches the expert label.
2. `supported_claims`: all substantive claims are supported by the fixed context.
3. `unsupported_claims`: number of substantive claims not supported by the context.
4. `key_evidence_omission`: output misses a material finding needed to justify the conclusion.
5. `misinterpretation`: output reverses or distorts a finding, direction, comparison, population, or uncertainty.
6. `contradiction`: output conflicts with explicit supplied evidence.
7. `incomplete_rationale`: label may be correct but the explanation omits a necessary reason.
8. `abstention_error`: output abstains despite a `SUFFICIENT` evidence label, or answers when the context is judged not sufficient (the latter is a secondary check outside the main subset).

Do not use exact string matching alone for rationale correctness. Use a written annotation guide and keep decisions traceable to context blocks. For this small pilot, manual claim-level review is acceptable; automated metrics can be added only after the rubric works.

## 7. Feasibility decision rules

These thresholds are pragmatic project gates, not established scientific constants:

- **Continue with PQA-L:** at least 20/30 items are clearly `SUFFICIENT`, their supporting evidence can be pointed to in the abstract context, and both prompt conditions can be run on exactly the same context.
- **Revise sampling/annotation:** 10–19 items are sufficient, or several labels are ambiguous. Expand the pilot modestly or seek second-annotator adjudication before changing research scope.
- **Do not lock PQA-L as the final dataset yet:** fewer than 10/30 are clearly sufficient, or evidence sufficiency cannot be distinguished from answer-key knowledge. Inspect another small set or consider BioASQ Phase B data, whose official task provides gold documents/snippets and exact/ideal answers; note that some BioASQ download access requires registration.

Even if the pilot passes, it proves feasibility only. It does not clear novelty, validate clinical safety, or provide statistically powered research results.

## 8. Important limitations

- PubMedQA is biomedical research QA based on literature abstracts, not patient-specific clinical advice, diagnosis, or longitudinal EHR reasoning.
- Its expert gold labels and abstract context do not guarantee that every provided context is independently sufficient.
- A single-student sufficiency label is not independent expert validation.
- A 30-item pilot is too small to establish performance improvements or generalizable error rates.
- Kim et al. and StagRAG remain direct novelty risks; StagRAG is unresolved and non-blocking.

## 9. Immediate steps

1. Download the official `ori_pqal.json` from https://github.com/pubmedqa/pubmedqa/tree/master/data, or run `prepare_pubmedqa_pilot.py` on a computer with internet access.
2. Annotate `pubmedqa_pilot_30_blinded.csv` before opening the key file.
3. Record the number of `SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`, `CONFLICTING`, and `AMBIGUOUS_OR_INVALID` cases.
4. Only then reveal the gold fields and run the two frozen prompting conditions.
5. Return the annotated CSV and key (or a safely merged pilot table) so the result can be reviewed and recorded in the project matrix.
