from types import SimpleNamespace

import pytest

from ai_healthcare.evaluation.multidomain_retrieval import (
    aggregate_metrics,
    evaluate_ranked_results,
)


def hit(document_id, domain):
    return SimpleNamespace(document_id=document_id, medical_domain=domain)


def test_expected_document_at_rank_two_has_mrr_half():
    metrics = evaluate_ranked_results(
        [hit("doc_other", "respiratory"), hit("doc_expected", "renal")],
        expected_document_ids=["doc_expected"],
        expected_domain="renal",
        k=5,
    )

    assert metrics.expected_rank == 2
    assert metrics.hit_at_1 is False
    assert metrics.hit_at_3 is True
    assert metrics.hit_at_k is True
    assert metrics.reciprocal_rank_at_k == 0.5
    assert metrics.top1_domain_match is False


def test_no_results_has_zero_hits_and_zero_mrr():
    metrics = evaluate_ranked_results(
        [],
        expected_document_ids=["doc_expected"],
        expected_domain="renal",
        k=5,
    )

    assert metrics.expected_rank is None
    assert metrics.hit_at_1 is False
    assert metrics.hit_at_3 is False
    assert metrics.hit_at_k is False
    assert metrics.reciprocal_rank_at_k == 0.0
    assert metrics.top1_domain_match is None


def test_domain_filtered_mode_requires_expected_domain():
    metrics = evaluate_ranked_results(
        [hit("doc_expected", "respiratory")],
        expected_document_ids=["doc_expected"],
        expected_domain="renal",
        k=5,
        require_domain_match=True,
    )

    assert metrics.expected_rank is None
    assert metrics.domain_filter_leakage == 1


def test_domain_leakage_is_counted_in_top_k():
    metrics = evaluate_ranked_results(
        [hit("doc_a", "renal"), hit("doc_b", "respiratory"), hit("doc_c", "renal")],
        expected_document_ids=["doc_c"],
        expected_domain="renal",
        k=3,
    )

    assert metrics.domain_filter_leakage == 1
    assert metrics.expected_rank == 3


def test_aggregate_metrics_preserves_query_denominator():
    first = evaluate_ranked_results(
        [hit("doc_expected", "renal")],
        expected_document_ids=["doc_expected"],
        expected_domain="renal",
        k=5,
    )
    second = evaluate_ranked_results(
        [hit("doc_other", "respiratory")],
        expected_document_ids=["doc_expected"],
        expected_domain="renal",
        k=5,
    )

    summary = aggregate_metrics([first, second])
    assert summary["query_count"] == 2
    assert summary["hit_rate_at_1"] == 0.5
    assert summary["hit_rate_at_k"] == 0.5
    assert summary["mrr_at_k"] == 0.5
    assert summary["top1_domain_accuracy"] == 0.5


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_k_must_be_positive_integer(bad_k):
    with pytest.raises(ValueError):
        evaluate_ranked_results(
            [],
            expected_document_ids=["doc_expected"],
            expected_domain="renal",
            k=bad_k,
        )
