"""Small, deterministic metrics for the MedlinePlus multi-domain retrieval pilot.

This module evaluates retrieval ranking only. It deliberately does not infer
answerability, medical correctness, or evidence sufficiency from similarity scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    expected_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_k: bool
    reciprocal_rank_at_k: float
    top1_domain_match: bool | None
    domain_filter_leakage: int


def evaluate_ranked_results(
    results: Sequence[Any],
    *,
    expected_document_ids: Sequence[str],
    expected_domain: str,
    k: int = 5,
    require_domain_match: bool = False,
) -> RetrievalCaseMetrics:
    """Score a ranked result list against the expected document(s).

    When require_domain_match=True, a hit requires both expected document ID
    and expected domain to match. This is useful for auditing filter behavior.
    """
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("results must be a sequence.")
    if isinstance(expected_document_ids, (str, bytes)) or not isinstance(
        expected_document_ids, Sequence
    ):
        raise TypeError("expected_document_ids must be a sequence of strings.")
    if not expected_document_ids:
        raise ValueError("expected_document_ids must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in expected_document_ids):
        raise ValueError("expected_document_ids must contain non-empty strings.")
    if not isinstance(expected_domain, str) or not expected_domain.strip():
        raise ValueError("expected_domain must be a non-empty string.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    expected_ids = set(expected_document_ids)
    ranked = list(results[:k])
    expected_rank: int | None = None
    domain_filter_leakage = 0

    for rank, result in enumerate(ranked, start=1):
        document_id = getattr(result, "document_id", None)
        medical_domain = getattr(result, "medical_domain", None)
        if not isinstance(document_id, str) or not isinstance(medical_domain, str):
            raise ValueError("Every result must expose document_id and medical_domain strings.")
        if medical_domain != expected_domain:
            domain_filter_leakage += 1

        is_expected = document_id in expected_ids
        if require_domain_match:
            is_expected = is_expected and medical_domain == expected_domain
        if is_expected and expected_rank is None:
            expected_rank = rank

    top1_domain_match: bool | None
    if results:
        top1_domain = getattr(results[0], "medical_domain", None)
        if not isinstance(top1_domain, str):
            raise ValueError("Top result must expose a medical_domain string.")
        top1_domain_match = top1_domain == expected_domain
    else:
        top1_domain_match = None

    return RetrievalCaseMetrics(
        expected_rank=expected_rank,
        hit_at_1=expected_rank == 1,
        hit_at_3=expected_rank is not None and expected_rank <= min(3, k),
        hit_at_k=expected_rank is not None,
        reciprocal_rank_at_k=(1.0 / expected_rank) if expected_rank is not None else 0.0,
        top1_domain_match=top1_domain_match,
        domain_filter_leakage=domain_filter_leakage,
    )


def aggregate_metrics(cases: Sequence[RetrievalCaseMetrics]) -> dict[str, int | float | None]:
    """Aggregate per-query retrieval metrics without hiding the denominator."""
    if not cases:
        return {
            "query_count": 0,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_k": 0,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_3": 0.0,
            "hit_rate_at_k": 0.0,
            "mrr_at_k": 0.0,
            "top1_domain_accuracy": None,
            "domain_filter_leakage": 0,
        }

    count = len(cases)
    domain_values = [case.top1_domain_match for case in cases if case.top1_domain_match is not None]
    return {
        "query_count": count,
        "hit_at_1": sum(case.hit_at_1 for case in cases),
        "hit_at_3": sum(case.hit_at_3 for case in cases),
        "hit_at_k": sum(case.hit_at_k for case in cases),
        "hit_rate_at_1": sum(case.hit_at_1 for case in cases) / count,
        "hit_rate_at_3": sum(case.hit_at_3 for case in cases) / count,
        "hit_rate_at_k": sum(case.hit_at_k for case in cases) / count,
        "mrr_at_k": sum(case.reciprocal_rank_at_k for case in cases) / count,
        "top1_domain_accuracy": (
            sum(domain_values) / len(domain_values) if domain_values else None
        ),
        "domain_filter_leakage": sum(case.domain_filter_leakage for case in cases),
    }
