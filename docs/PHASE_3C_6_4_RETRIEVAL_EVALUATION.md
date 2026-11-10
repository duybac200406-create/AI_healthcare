# Phase 3C.6.4 – Evaluation protocol

## Benchmark integrity

The benchmark contains 28 answerable pilot queries (four per domain) and seven questions marked out-of-scope for this pilot. The Type 2 Diabetes expected document identifier has been corrected to match the source manifest and indexed document: `doc_medlineplus_type2_diabetes`.

The runner fails before making embedding requests if an expected ID is absent from the manifest or the expected domain does not match the manifest's domain.

## Primary evaluation: no domain filter

This is the more realistic dense-retrieval baseline because the query arrives without a known domain label. Hit@1/3/5 and MRR@5 measure whether the annotated expected source appears in the ranked list. Top-1 domain accuracy measures whether the first result is in the annotated domain.

## Diagnostic evaluation: known-domain filter

The runner repeats each answerable query with the gold domain passed as a metadata filter. This measures within-domain retrieval under an oracle filter; it is not evidence that a future Commander can classify domains correctly.

## Out-of-scope queries

The seven out-of-scope questions are sent to retrieval and nearest neighbors are recorded. Similarity scores are not converted into an answerable/unanswerable decision. A separate agent-level evaluation must test abstention with evidence that is irrelevant or insufficient.

## Known limitations

- One summary per domain means limited document diversity.
- The expected source labels are document-level; retrieval may return the expected document but the wrong chunk.
- The current benchmark does not automatically verify coverage of every expected evidence concept in the retrieved chunk.
- No Qwen answer generation is included in this phase.
- This is an engineering pilot, not a clinical validation study.
