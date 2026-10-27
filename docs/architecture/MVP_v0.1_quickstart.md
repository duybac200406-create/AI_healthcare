# AI Healthcare — MVP v0.1

A deliberately small prototype for validating patient-record fidelity before integrating an LLM or a medical RAG pipeline.

## Scope

- Uses synthetic patient data only.
- Demonstrates exact handling of `RECORDED_AS_CONFIRMED` and `NOT_ASSESSED`.
- Shows fixed evidence excerpts and citations `[E1]`, `[E2]`.
- Does not diagnose, provide personalized treatment, call an LLM, or retrieve from an external medical corpus.

## Run in PowerShell

```powershell
cd "D:\AI Healthcare"
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

## Safety boundary

This is a software prototype using synthetic data, not a clinically validated medical product. Do not place real patient data in this project.
