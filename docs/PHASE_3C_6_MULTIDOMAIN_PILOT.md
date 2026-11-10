# Phase 3C.6 — Multi-domain Medical Corpus Pilot v0.1

## Purpose

Build a small, auditable corpus to test whether the existing Medical RAG pipeline retrieves and cites information across several medical domains. This is not a comprehensive medical knowledge base, clinical guideline repository, diagnostic system, or quality certification.

## Pilot scope

Seven MedlinePlus Health Topic pages:

1. Cardiovascular — Heart Diseases
2. Endocrine/metabolic — Type 2 Diabetes
3. Respiratory — COPD
4. Gastrointestinal — GERD
5. Renal — Chronic Kidney Disease
6. Neurology — Migraine
7. Mental health — Depression

The source list and displayed page update dates are recorded in `data/manifests/medlineplus_multidomain_pilot_v0.1.json`.

## Copyright and scope policy

Only ingest the `Summary` section of each MedlinePlus Health Topic page. NLM says Health Topic summaries are public domain and asks users to acknowledge MedlinePlus. Do not ingest A.D.A.M. Medical Encyclopedia material, AHFS drug monographs, third-party linked material, PDFs, images, or videos under this policy. Check the official MedlinePlus usage policy before expanding the source scope:

- https://medlineplus.gov/about/using/usingcontent/
- https://medlineplus.gov/about/general/reviewandupdate/

Attribution: `Source: MedlinePlus, National Library of Medicine.`

The page's `Last updated` value is stored as `source_last_updated`, not `publication_date`. NLM says that for Health Topics it indicates when the topic summary and/or page links were reviewed and updated; it does not prove that every sentence was clinically reviewed on that date.

## Metadata requirements before ingestion

The runtime schema should support:

- `source_last_updated: date | None`
- a controlled `medical_domain` primary tag
- a dedicated `health_topic_summary` source type
- domain filtering in VectorSearchFilter and Qdrant payload

Existing fields retain their meaning:

- `publication_date`: only a true publication date, otherwise null
- `retrieved_at`: the time our pipeline fetched/captured the source
- `license`: a concise rights/provenance note
- `source_url`: the exact page URL

Domain labels are curated metadata, not diagnoses or claims made by the source.

## Evaluation design

The evaluation JSON contains 28 answerable questions (four per domain) and seven questions explicitly unanswerable from this pilot. Expected evidence concepts are evaluation targets, not generated answers.

Score separately:

1. Retrieval recall@k and correct-domain retrieval.
2. Evidence support for each generated claim.
3. Citation correctness and provenance integrity.
4. Abstention on questions not supported by the pilot corpus.
5. Cross-domain retrieval behavior and false-positive retrieval.

Do not interpret cosine similarity as medical confidence or evidence sufficiency. A top-k hit alone does not make a question answerable.

## Limitations

- Seven documents are a breadth pilot, not comprehensive coverage.
- All seven are from one source platform/institution family; this tests topic breadth, not independent source triangulation.
- The summaries are consumer health overviews, not specialty clinical practice guidelines.
- Medication dosage, interactions, and treatment selection are deliberately outside this corpus.
- English source text is retained; answer generation may be in Vietnamese, but translations must not be substituted as source text without provenance.

## Next steps

1. Add provenance fields and domain taxonomy to `MedicalDocument` and `VectorSearchResult`.
2. Extend Qdrant payload and allowlisted filters with `medical_domain` and `source_last_updated`.
3. Capture only the Summary text for the seven pages and validate metadata against the manifest.
4. Ingest to a dedicated pilot collection, not the production-intended medical collection.
5. Run the 35-query benchmark and review failures by domain and question angle.
6. Add independent, properly licensed sources only after separate source and license review.
