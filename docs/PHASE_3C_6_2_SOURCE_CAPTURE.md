# Phase 3C.6.2 – MedlinePlus Source Capture

## Scope

Capture only the `FullSummary` field from exact MedlinePlus English Health Topic records via the official MedlinePlus Web Service. The capture code verifies both the exact topic URL and exact title/organization returned by the service. It does not scrape the whole HTML page and does not follow linked resources.

The current manifest is `data/manifests/medlineplus_multidomain_pilot_v0.2.json`. It intentionally corrects the Type 2 Diabetes topic URL to `https://medlineplus.gov/diabetestype2.html` and uses the page title `Diabetes Type 2`.

## Source policy

MedlinePlus says Health Topic page summaries are public domain; other material such as A.D.A.M. Medical Encyclopedia and AHFS drug monographs is copyrighted/licensed. The capture path requests `rettype=brief` and extracts only `content name="FullSummary"`. Attribution is retained. No images, external links, encyclopedia content, drug monographs, or third-party article text are captured.

Official documentation:
- Web Service: https://medlineplus.gov/about/developers/webservices/
- Content use: https://medlineplus.gov/about/using/usingcontent/
- Review/update dates: https://medlineplus.gov/about/general/reviewandupdate/

MedlinePlus states the Web Service is updated daily Tuesday–Saturday, no more than 85 requests/minute per IP are allowed, and caching for 12–24 hours is recommended. The script fetches one request per topic sequentially and defaults to a one-second delay. It caches successful captures locally unless `--refresh` is specified.

## Dates

`source_last_updated` in the manifest is manually verified against the official topic-page footer as of 2026-11-08. The Web Service response is not treated as the authority for this page-footer date. MedlinePlus explains that the date indicates when the topic summary and/or page links were reviewed and updated; it is not a publication date. `retrieved_at` is generated at capture time and stored separately. `publication_date` remains null.

## Run

From project root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_medlineplus_capture.py -v
.\.venv\Scripts\python.exe scripts/capture_medlineplus_multidomain.py
```

Use `--refresh` to fetch all seven records again. Do not use `--refresh` on every run; observe the recommended 12–24 hour cache window. The output directory is `data/samples/medlineplus_health_topics/`.

## Validation rules

- Match exactly one document with the canonical expected MedlinePlus URL.
- Verify exact title and `National Library of Medicine` organization.
- Require a `FullSummary` node and non-trivial extracted content.
- Preserve paragraphs and bullet boundaries as plain text.
- Preserve `medical_domain`, `source_last_updated`, URL, source type, and attribution/license metadata.
- Fail rather than substitute a similar topic or a non-MedlinePlus page.

This is a small pilot source capture step, not a comprehensive or clinically validated knowledge base.
