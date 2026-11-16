from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_healthcare.evaluation.benchmarks.cadec_v2 import CadecV2Adapter  # noqa: E402
from ai_healthcare.providers.base import GenerationConfig  # noqa: E402
from ai_healthcare.providers.ollama_provider import (  # noqa: E402
    OllamaProvider,
    OllamaProviderError,
)

DEFAULT_DATA_ROOT = Path(
    os.environ.get("CADEC_V2_DATA", r"D:\datasets\CADECv2\extracted\raw\data")
)
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
PROMPT_PATH = ROOT / "prompts" / "cadec_v2_symptom_span_en.md"
EXPECTED_DEV_DOCUMENTS = 354
EXPECTED_DEV_GOLD_SPANS = 230


class PredictedMention(BaseModel):
    """Benchmark-only mention contract; the model supplies text and context, not offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=180)
    left_context: str = Field(max_length=80)
    right_context: str = Field(max_length=80)


class CadecSpanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mentions: list[PredictedMention] = Field(max_length=100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CADECv2 Dev zero-shot evaluation with deterministic context alignment."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional smoke-test limit. Any limited run is marked partial, not a full benchmark.",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run dataset, prompt, Ollama, and model checks without generating any model output.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Test the deterministic text/context locator only; no medical data or LLM calls.",
    )
    parser.add_argument("--verbose-debug", action="store_true")
    return parser.parse_args()


def build_system_prompt() -> str:
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Benchmark prompt not found: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    schema = json.dumps(CadecSpanOutput.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    return f"{prompt}\n\n## Required JSON Schema\n\n{schema}\n"


def print_check(level: str, name: str, detail: str) -> None:
    print(f"[{level}] {name}: {detail}")


def find_all_occurrences(source: str, needle: str) -> list[int]:
    """Find exact, potentially overlapping occurrences without text normalization."""
    if not needle:
        return []
    positions: list[int] = []
    cursor = 0
    while True:
        position = source.find(needle, cursor)
        if position < 0:
            break
        positions.append(position)
        cursor = position + 1
    return positions


def _left_context_matches(source: str, start: int, context: str) -> bool:
    if not context:
        return start == 0
    return source[:start].endswith(context)


def _right_context_matches(source: str, end: int, context: str) -> bool:
    if not context:
        return end == len(source)
    return source[end:].startswith(context)


def resolve_mention_offset(mention: PredictedMention, source: str) -> dict[str, Any]:
    """Resolve an exact mention string; use adjacent context only for repeated surfaces."""
    occurrences = find_all_occurrences(source, mention.text)
    if not occurrences:
        return {
            "resolved": False,
            "resolution_status": "text_not_found_exactly",
            "start_char": None,
            "end_char": None,
            "occurrence_count": 0,
        }

    if len(occurrences) == 1:
        start = occurrences[0]
        end = start + len(mention.text)
        return {
            "resolved": True,
            "resolution_status": "unique_exact_text",
            "start_char": start,
            "end_char": end,
            "occurrence_count": 1,
        }

    context_matches = []
    for start in occurrences:
        end = start + len(mention.text)
        if (
            _left_context_matches(source, start, mention.left_context)
            and _right_context_matches(source, end, mention.right_context)
        ):
            context_matches.append(start)

    if len(context_matches) == 1:
        start = context_matches[0]
        return {
            "resolved": True,
            "resolution_status": "repeated_text_disambiguated_by_context",
            "start_char": start,
            "end_char": start + len(mention.text),
            "occurrence_count": len(occurrences),
        }

    return {
        "resolved": False,
        "resolution_status": (
            "ambiguous_context" if len(context_matches) > 1 else "context_mismatch"
        ),
        "start_char": None,
        "end_char": None,
        "occurrence_count": len(occurrences),
        "context_candidate_count": len(context_matches),
    }


def align_predictions(parsed: CadecSpanOutput, source: str) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for mention in parsed.mentions:
        resolution = resolve_mention_offset(mention, source)
        start = resolution["start_char"]
        end = resolution["end_char"]
        offset_valid = (
            resolution["resolved"]
            and isinstance(start, int)
            and isinstance(end, int)
            and source[start:end] == mention.text
        )
        aligned.append(
            {
                "text": mention.text,
                "left_context": mention.left_context,
                "right_context": mention.right_context,
                "start_char": start if offset_valid else None,
                "end_char": end if offset_valid else None,
                "offset_valid": bool(offset_valid),
                "resolution_status": resolution["resolution_status"],
                "surface_occurrence_count": resolution["occurrence_count"],
            }
        )
    return aligned


def metric_counts(gold_offsets: list[tuple[int, int]], predictions: list[dict[str, Any]]) -> dict[str, int | float]:
    gold_counter = Counter(gold_offsets)
    predicted_valid = Counter(
        (item["start_char"], item["end_char"])
        for item in predictions
        if item["offset_valid"]
    )
    tp = sum((gold_counter & predicted_valid).values())
    predicted_count = len(predictions)
    gold_count = len(gold_offsets)
    fp = predicted_count - tp  # unresolved and duplicate predictions remain false positives
    fn = gold_count - tp
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "gold": gold_count,
        "predicted": predicted_count, "precision": precision,
        "recall": recall, "f1": f1,
    }


def aggregate_document_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, int | float]:
    """
    Aggregate metrics by document.

    Never compare character offsets across different documents.
    Recompute each document's metrics independently, then sum counts.
    """
    tp = 0
    fp = 0
    fn = 0
    gold = 0
    predicted = 0

    for row in rows:
        gold_offsets = [
            (span["start_char"], span["end_char"])
            for span in row["gold_spans"]
        ]

        # Each document is evaluated independently.
        metrics = metric_counts(
            gold_offsets,
            row.get("predictions", []),
        )

        tp += metrics["tp"]
        fp += metrics["fp"]
        fn += metrics["fn"]
        gold += metrics["gold"]
        predicted += metrics["predicted"]

    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold": gold,
        "predicted": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def surface_match_diagnostics(gold_texts: list[str], predictions: list[dict[str, Any]]) -> dict[str, int]:
    gold_exact = Counter(gold_texts)
    pred_exact = Counter(item["text"] for item in predictions)
    gold_folded = Counter(value.casefold() for value in gold_texts)
    pred_folded = Counter(item["text"].casefold() for item in predictions)
    return {
        "exact_surface_matches": sum((gold_exact & pred_exact).values()),
        "case_insensitive_surface_matches": sum((gold_folded & pred_folded).values()),
    }


def _ollama_model_check(base_url: str, model_name: str, timeout: float) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        available = {
            item.get("name") or item.get("model")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        if model_name in available or f"{model_name}:latest" in available:
            return True, f"Ollama reachable; model {model_name} is installed"
        return False, f"Ollama reachable, but {model_name} not found; available={sorted(x for x in available if x)}"
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return False, f"Cannot verify Ollama/model: {type(exc).__name__}: {exc}"


def debug_check(
    *,
    data_root: Path,
    ollama_url: str,
    model: str,
    timeout: float,
    num_predict: int,
    output_root: Path,
    documents: list[Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    documents_by_id: dict[str, Any] | None = None,
    expected_documents: int | None = None,
) -> dict[str, Any]:
    """Preflight and post-run diagnostics; reports errors without exposing prompt/text content."""
    checks: list[dict[str, str]] = []

    def add(level: str, name: str, detail: str) -> None:
        checks.append({"level": level, "name": name, "detail": detail})
        print_check(level, name, detail)

    required_dirs = [data_root / "txt", data_root / "ann", data_root / "split"]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    add("PASS" if not missing_dirs else "FAIL", "dataset_directories", "found txt/ann/split" if not missing_dirs else f"missing: {missing_dirs}")
    add("PASS" if PROMPT_PATH.is_file() else "FAIL", "english_prompt", str(PROMPT_PATH) if PROMPT_PATH.is_file() else "prompt file missing")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".cadec_debug_write_check.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("PASS", "output_directory", f"writable: {output_root}")
    except OSError as exc:
        add("FAIL", "output_directory", f"not writable: {type(exc).__name__}: {exc}")

    model_ok, model_detail = _ollama_model_check(ollama_url, model, min(timeout, 8.0))
    add("PASS" if model_ok else "FAIL", "ollama_model", model_detail)
    add("PASS" if num_predict >= 2048 else "WARN", "output_token_budget", f"num_predict={num_predict}; 2048 or higher is recommended for this structured output")

    if documents is not None:
        lengths = [len(doc.text) for doc in documents]
        gold_count = sum(len(doc.mentions) for doc in documents)
        negative_count = sum(not doc.mentions for doc in documents)
        limit_ok = not lengths or max(lengths) <= 4000
        add("PASS" if limit_ok else "FAIL", "message_length", f"max={max(lengths, default=0)} chars; agent boundary=4000")
        if expected_documents is not None:
            same = len(documents) == expected_documents
            add("PASS" if same else "WARN", "document_coverage", f"loaded={len(documents)} expected={expected_documents}")
        add("PASS", "gold_and_negative_coverage", f"gold={gold_count}; continuous-negative docs={negative_count}")

    if rows is not None:
        status_counts = Counter(row["status"] for row in rows)
        all_predictions = [pred for row in rows for pred in row.get("predictions", [])]
        unresolved = sum(not pred["offset_valid"] for pred in all_predictions)
        resolved = len(all_predictions) - unresolved
        model_errors = status_counts.get("model_output_error", 0)
        infrastructure_errors = status_counts.get("infrastructure_error", 0)
        count_ok = expected_documents is None or len(rows) == expected_documents
        add("PASS" if count_ok else "FAIL", "processed_documents", f"processed={len(rows)} expected={expected_documents}")
        add("PASS" if infrastructure_errors == 0 else "FAIL", "infrastructure_errors", str(infrastructure_errors))
        add("PASS" if model_errors == 0 else "WARN", "structured_output_errors", str(model_errors))
        add("PASS" if unresolved == 0 else "WARN", "span_alignment", f"resolved={resolved}; unresolved={unresolved}")
        invariant_errors = []
        if documents_by_id is not None:
            for row in rows:
                source_doc = documents_by_id.get(row["document_id"])
                if source_doc is None:
                    invariant_errors.append(f"missing source document: {row['document_id']}")
                    continue
                for pred in row.get("predictions", []):
                    if not pred.get("offset_valid"):
                        continue
                    start, end = pred.get("start_char"), pred.get("end_char")
                    if not isinstance(start, int) or not isinstance(end, int):
                        invariant_errors.append(f"missing offsets: {row['document_id']}")
                    elif source_doc.text[start:end] != pred.get("text"):
                        invariant_errors.append(f"source mismatch: {row['document_id']}:{start}:{end}")
        else:
            invariant_errors = ["source documents not supplied for post-run verification"]
        add("PASS" if not invariant_errors else "FAIL", "resolved_offset_invariant", "all resolved spans match original source" if not invariant_errors else "; ".join(invariant_errors[:5]))

    fail_count = sum(item["level"] == "FAIL" for item in checks)
    warn_count = sum(item["level"] == "WARN" for item in checks)
    return {"checks": checks, "fail_count": fail_count, "warning_count": warn_count, "passed": fail_count == 0}


def run_locator_self_test() -> int:
    """Technical unit checks for the offset resolver; these are not symptom examples or benchmark data."""
    source = "Opening line. chest pain after dose; later, chest pain returned."
    first = PredictedMention(text="chest pain", left_context="Opening line. ", right_context=" after dose; later, chest pain returned.")
    second = PredictedMention(text="chest pain", left_context="after dose; later, ", right_context=" returned.")
    ambiguous = PredictedMention(text="chest pain", left_context="", right_context="")
    unique = PredictedMention(text="Opening line", left_context="", right_context=". chest pain after dose; later, chest pain returned.")
    missing = PredictedMention(text="headache", left_context="", right_context="")

    checks = [
        ("first repeated occurrence", resolve_mention_offset(first, source)["start_char"] == source.index("chest pain")),
        ("second repeated occurrence", resolve_mention_offset(second, source)["start_char"] == source.rindex("chest pain")),
        ("ambiguous repeated occurrence", not resolve_mention_offset(ambiguous, source)["resolved"]),
        ("unique exact occurrence", resolve_mention_offset(unique, source)["resolved"]),
        ("missing text", not resolve_mention_offset(missing, source)["resolved"]),
    ]
    failures = 0
    print("========== LOCATOR SELF-TEST (technical only) ==========")
    for name, passed in checks:
        print_check("PASS" if passed else "FAIL", name, "ok" if passed else "unexpected locator result")
        failures += not passed
    print(f"Self-test: {len(checks) - failures}/{len(checks)} passed")
    return 0 if failures == 0 else 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_locator_self_test()
    if args.timeout <= 0 or args.num_ctx <= 0 or args.num_predict <= 0:
        print_check("FAIL", "arguments", "timeout, num-ctx and num-predict must be positive")
        return 1
    if args.limit is not None and args.limit <= 0:
        print_check("FAIL", "limit", "--limit must be a positive integer")
        return 1

    try:
        adapter = CadecV2Adapter(args.data_root)
        split_validation = adapter.validate_splits()
        if not split_validation.get("valid"):
            print_check("FAIL", "official_splits", f"invalid: {split_validation}")
            return 1
        all_documents = adapter.load_split("dev", labels={"Symptom"}, include_discontinuous=False)
        if not all_documents:
            print_check("FAIL", "dev_documents", "CADECv2 Dev is empty")
            return 1
        if args.limit is not None:
            documents = all_documents[:args.limit]
            expected_for_run = len(documents)
            print_check("WARN", "partial_run", f"smoke test only: {len(documents)}/{len(all_documents)} Dev documents")
        else:
            documents = all_documents
            expected_for_run = len(all_documents)
    except Exception as exc:
        print_check("FAIL", "dataset_loading", f"{type(exc).__name__}: {exc}")
        return 1

    expected_audit_count = args.limit is None
    expected_count = EXPECTED_DEV_DOCUMENTS if expected_audit_count else expected_for_run
    debug = debug_check(
        data_root=args.data_root,
        ollama_url=args.ollama_url,
        model=args.model,
        timeout=args.timeout,
        num_predict=args.num_predict,
        output_root=args.output_root,
        documents=documents,
        expected_documents=expected_count,
    )
    if not debug["passed"]:
        print_check("FAIL", "preflight", "fix the failed checks before running inference")
        return 1
    if args.preflight_only:
        print("Preflight complete. No model inference was performed.")
        return 0

    if expected_audit_count and len(documents) != EXPECTED_DEV_DOCUMENTS:
        print_check("WARN", "audited_dev_size", f"current={len(documents)}; previous audited snapshot={EXPECTED_DEV_DOCUMENTS}")

    try:
        system_prompt = build_system_prompt()
    except Exception as exc:
        print_check("FAIL", "prompt_loading", f"{type(exc).__name__}: {exc}")
        return 1

    gold_total = sum(len(doc.mentions) for doc in documents)
    negative_documents = sum(not doc.mentions for doc in documents)
    config = GenerationConfig(temperature=args.temperature, num_ctx=args.num_ctx, num_predict=args.num_predict)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"cadec_v2_dev_context_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    debug_path = run_dir / "debug_report.json"

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print("\n========== CADECv2 ZERO-SHOT BASELINE — CONTEXT ALIGNMENT ==========")
    print(f"Split: dev | Documents: {len(documents)} | Gold spans: {gold_total}")
    print(f"Continuous-negative documents: {negative_documents}")
    print(f"Model: {args.model} | temperature={args.temperature} | num_predict={args.num_predict}")
    print(f"Output: {run_dir}")

    with OllamaProvider(
        base_url=args.ollama_url,
        model=args.model,
        timeout=args.timeout,
        no_think=True,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    ) as provider, predictions_path.open("w", encoding="utf-8", newline="\n") as out_file:
        for index, document in enumerate(documents, start=1):
            gold_spans = [(m.segments[0].start, m.segments[0].end) for m in document.mentions]
            gold_texts = [m.segments[0].text for m in document.mentions]
            user_prompt = json.dumps({"document_text": document.text}, ensure_ascii=False, separators=(",", ":"))
            status = "ok"
            predictions: list[dict[str, Any]] = []
            error: dict[str, Any] | None = None
            raw_chars: int | None = None

            for attempt in range(2):
                try:
                    raw_output = provider.generate_json(
                        system_prompt,
                        user_prompt,
                        CadecSpanOutput.model_json_schema(),
                        config=config,
                    )
                    raw_chars = len(raw_output)
                    parsed = CadecSpanOutput.model_validate_json(raw_output)
                    predictions = align_predictions(parsed, document.text)
                    break
                except OllamaProviderError as exc:
                    if attempt == 0:
                        continue
                    status = "infrastructure_error"
                    error = {"type": type(exc).__name__, "message": str(exc)[:300]}
                except (ValidationError, ValueError, TypeError) as exc:
                    status = "model_output_error"
                    validation_errors = exc.errors(include_url=False) if isinstance(exc, ValidationError) else []
                    eof_suspected = any("EOF" in item.get("msg", "") for item in validation_errors)
                    error = {
                        "type": type(exc).__name__,
                        "issues": [
                            {"loc": list(item.get("loc", ())), "type": item.get("type"), "message": item.get("msg")}
                            for item in validation_errors[:10]
                        ],
                        "raw_output_chars": raw_chars,
                        "truncation_suspected": eof_suspected,
                    }
                    break

            metrics = metric_counts(gold_spans, predictions)
            surface_diagnostics = surface_match_diagnostics(gold_texts, predictions)
            resolution_counts = Counter(item["resolution_status"] for item in predictions)
            row = {
                "document_id": document.document_id,
                "status": status,
                "gold_spans": [
                    {"start_char": m.segments[0].start, "end_char": m.segments[0].end, "text": m.segments[0].text}
                    for m in document.mentions
                ],
                "predictions": predictions,
                "surface_match_diagnostics": surface_diagnostics,
                "resolution_counts": dict(resolution_counts),
                "metrics": metrics,
                "raw_output_chars": raw_chars,
                "error": error,
            }
            rows.append(row)
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()

            status_label = "WARN" if status != "ok" else "INFO"
            unresolved = sum(not item["offset_valid"] for item in predictions)
            print(
                f"[{index:03d}/{len(documents)}] {document.document_id} status={status} "
                f"gold={metrics['gold']} pred={metrics['predicted']} tp={metrics['tp']} "
                f"fp={metrics['fp']} fn={metrics['fn']} unresolved={unresolved}"
            )
            if args.verbose_debug and error:
                print_check(status_label, f"{document.document_id}_error", json.dumps(error, ensure_ascii=False))

    elapsed = time.perf_counter() - started
    totals = aggregate_document_metrics(rows)
    status_counts = Counter(row["status"] for row in rows)
    model_error_count = status_counts.get("model_output_error", 0)
    infrastructure_error_count = status_counts.get("infrastructure_error", 0)
    all_predictions = [p for row in rows for p in row["predictions"]]
    resolution_counts = Counter(p["resolution_status"] for p in all_predictions)
    total_surface_exact = sum(row["surface_match_diagnostics"]["exact_surface_matches"] for row in rows)
    total_surface_folded = sum(row["surface_match_diagnostics"]["case_insensitive_surface_matches"] for row in rows)
    complete_document_run = len(rows) == len(documents)
    full_dev_run = args.limit is None and len(documents) == len(all_documents)
    benchmark_complete = complete_document_run and infrastructure_error_count == 0
    output_valid = model_error_count == 0

    summary = {
        "benchmark": "CADECv2",
        "track": "continuous Symptom span extraction",
        "split": "dev",
        "run_scope": "full_official_dev" if full_dev_run else "partial_smoke_test",
        "benchmark_complete": benchmark_complete,
        "all_model_outputs_valid": output_valid,
        "documents_expected_for_run": len(documents),
        "documents_processed": len(rows),
        "gold_continuous_symptom_spans": gold_total,
        "continuous_negative_documents": negative_documents,
        "model_output_error_documents": model_error_count,
        "infrastructure_error_documents": infrastructure_error_count,
        "raw_exact_surface_matches": total_surface_exact,
        "raw_case_insensitive_surface_matches": total_surface_folded,
        "span_resolution_status_counts": dict(resolution_counts),
        "model": args.model,
        "generation_config": {
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "timeout_seconds": args.timeout,
            "no_think": True,
        },
        "scoring": "strict exact character offsets resolved by exact source text and adjacent contexts; unresolved and duplicate predictions count as false positives",
        "metrics": totals,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_path": str(PROMPT_PATH),
        "data_root": str(args.data_root),
        "split_validation": split_validation,
        "notes": [
            "The official Test split is not loaded or evaluated by this script.",
            "All loaded Dev documents are retained, including continuous-negative documents.",
            "Discontinuous Symptom annotations are excluded from this continuous-span track.",
            "The model returns source text and adjacent contexts; Python calculates offsets deterministically.",
            "An exact surface occurring once is aligned to its unique occurrence. Repeated surfaces require one unique exact context match.",
            "Unresolved mentions remain predictions and count as false positives; their missed Gold spans count as false negatives.",
            "Model-output/schema failures are recorded as zero predictions for the document; infrastructure failures make the run incomplete.",
        ],
    }
    write_json(summary_path, summary)

    post_debug = debug_check(
        data_root=args.data_root,
        ollama_url=args.ollama_url,
        model=args.model,
        timeout=args.timeout,
        num_predict=args.num_predict,
        output_root=args.output_root,
        rows=rows,
        documents_by_id={doc.document_id: doc for doc in documents},
        expected_documents=len(documents),
    )
    post_debug["summary_path"] = str(summary_path)
    write_json(debug_path, post_debug)

    print("\n========== SUMMARY ==========")
    print(f"Run complete: {complete_document_run} | Full official Dev: {full_dev_run}")
    print(f"Documents: {len(rows)}/{len(documents)} | TP={totals['tp']} FP={totals['fp']} FN={totals['fn']}")
    print(f"Precision={totals['precision']:.4f} Recall={totals['recall']:.4f} F1={totals['f1']:.4f}")
    print(f"Exact Gold surface matches (diagnostic only): {total_surface_exact}")
    print(f"Model output errors: {model_error_count} | Infrastructure errors: {infrastructure_error_count}")
    print(f"Debug report: {debug_path}")
    print(f"Summary: {summary_path}")
    print(f"Predictions: {predictions_path}")

    if infrastructure_error_count:
        return 2
    if model_error_count:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
