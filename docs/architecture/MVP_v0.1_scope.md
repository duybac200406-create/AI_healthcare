# AI Healthcare MVP v0.1 — Scope and boundaries

## Included
- Streamlit interface.
- One synthetic patient record (SYN-001).
- Fixed evidence excerpts E1 and E2.
- Pydantic validation of the patient schema.
- Deterministic response path that preserves `RECORDED_AS_CONFIRMED` and `NOT_ASSESSED`.
- Unit tests for status preservation and out-of-scope abstention.

## Not included yet
- LLM/API provider integration.
- Vector database, embeddings, or medical corpus ingestion.
- Real accounts or real patient data.
- Diagnosis, treatment recommendations, or clinical deployment.

## Why deterministic first?
This first slice validates the data path and critical status invariants before adding a generative model, where output variability introduces more failure modes.
