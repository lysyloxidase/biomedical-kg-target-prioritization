from __future__ import annotations

import json

import numpy as np
import pytest

from kgtp.eval.metrics import (
    auprc,
    auroc,
    brier_score,
    build_candidate_score_map,
    evaluate_binary_and_ranking,
    evaluate_full_candidate,
    evaluate_sampled_unlabeled,
    expected_calibration_error,
    filtered_ranks,
    hits_at_k,
    mrr,
    sample_negative_triples,
)
from kgtp.eval.runner import paired_significance, run_multiseed


def test_binary_and_rank_metrics_are_deterministic() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    ranks = np.asarray([1, 3, 10], dtype=float)

    assert auroc(labels, scores) == 0.75
    assert round(auprc(labels, scores), 3) == 0.833
    assert hits_at_k(ranks, 3) == 2 / 3
    assert round(mrr(ranks), 3) == round((1 + 1 / 3 + 1 / 10) / 3, 3)


def test_filtered_ranking_removes_other_known_true_tails() -> None:
    scores = {
        ("D1", "associated_with"): {
            "G1": 0.8,
            "G2": 0.9,
            "G3": 0.1,
        }
    }
    true = [("D1", "associated_with", "G1")]
    known = {("D1", "associated_with", "G1"), ("D1", "associated_with", "G2")}

    assert filtered_ranks(scores, true, known, filtered=False).tolist() == [2.0]
    assert filtered_ranks(scores, true, known, filtered=True).tolist() == [1.0]


def test_negative_sampling_uses_all_eligible_when_less_than_1000() -> None:
    positives = [("D1", "associated_with", "G1")]
    known = {("D1", "associated_with", "G1"), ("D1", "associated_with", "G2")}
    tails = {("D1", "associated_with"): ["G1", "G2", "G3", "G4"]}

    negatives = sample_negative_triples(
        positives,
        all_known=known,
        tail_candidates=tails,
        negatives_per_positive=1_000,
        seed=13,
    )

    assert set(negatives) == {
        ("D1", "associated_with", "G3"),
        ("D1", "associated_with", "G4"),
    }


def test_evaluate_binary_and_ranking_reports_raw_and_filtered_metrics() -> None:
    positives = [("D1", "associated_with", "G1")]
    known = {("D1", "associated_with", "G1"), ("D1", "associated_with", "G2")}
    tails = {("D1", "associated_with"): ["G1", "G2", "G3", "G4"]}
    score_values = {"G1": 0.8, "G2": 0.9, "G3": 0.2, "G4": 0.1}

    result = evaluate_binary_and_ranking(
        lambda triple: score_values[triple[2]],
        positives,
        all_known=known,
        tail_candidates=tails,
        negatives_per_positive=1_000,
        seed=13,
    )
    candidate_scores = build_candidate_score_map(
        lambda triple: score_values[triple[2]],
        positives,
        all_known=known,
        tail_candidates=tails,
        negatives_per_positive=1_000,
        seed=13,
    )

    assert result["primary_metric"] == "AUPRC"
    assert result["unlabeled_count"] == 2
    assert "AUROC" in result and "AUPRC" in result
    assert result["filtered"]["MRR"] == 1.0  # type: ignore[index]
    assert candidate_scores[("D1", "associated_with")]["G1"] == 0.8


def test_auprc_treats_all_tied_scores_as_prevalence() -> None:
    assert auprc([1, 1, 0, 0, 0], [0.0] * 5) == 0.4


def test_multiseed_runner_writes_mean_std_ci_and_significance(tmp_path) -> None:
    payload = run_multiseed(
        "toy",
        lambda seed: {"AUPRC": seed / 100, "AUROC": 0.9},
        seeds=[11, 13, 17, 19, 23],
        output_dir=tmp_path,
    )
    significance = paired_significance(
        payload["seed_results"],
        {seed: {"AUPRC": 0.1} for seed in [11, 13, 17, 19, 23]},
    )

    saved = json.loads((tmp_path / "results_toy.json").read_text(encoding="utf-8"))
    assert saved["primary_metric"] == "AUPRC"
    assert payload["summary"]["AUPRC"]["ci95"] > 0
    assert significance["test"] == "paired_t_test"
    assert "paired_bootstrap_95_ci" in significance
    sign_flip_raw = significance["paired_sign_flip_p_value"]
    assert isinstance(sign_flip_raw, float)
    sign_flip = sign_flip_raw
    assert 0.0 <= sign_flip <= 1.0


def test_full_candidate_metrics_include_topk_enrichment_and_calibration() -> None:
    positives = [
        ("D1", "associated_with", "G1"),
        ("D1", "associated_with", "G2"),
    ]
    known = {
        *positives,
        ("D1", "associated_with", "G3"),
    }
    tails = {
        ("D1", "associated_with"): ["G1", "G2", "G3", "G4", "G5"],
    }
    values = {"G1": 3.0, "G2": 2.0, "G3": 4.0, "G4": 1.0, "G5": 0.0}

    result = evaluate_full_candidate(
        lambda triple: values[triple[2]],
        positives,
        all_known=known,
        tail_candidates=tails,
        probability_scorer=lambda triple: 1 / (1 + np.exp(-values[triple[2]])),
    )

    assert result["candidate_count"] == 4
    assert result["candidate_prevalence"] == 0.5
    assert result["filtered"]["Precision@1"] == 1.0  # type: ignore[index]
    assert result["filtered"]["Recall@3"] == 1.0  # type: ignore[index]
    assert result["filtered"]["NDCG@3"] == 1.0  # type: ignore[index]
    assert result["filtered"]["EF@1"] == 2.0  # type: ignore[index]
    assert "Brier" in result["calibration"]  # type: ignore[operator]
    assert "ECE" in result["calibration"]  # type: ignore[operator]


def test_sampled_unlabeled_metrics_record_prevalence_and_semantics() -> None:
    result = evaluate_sampled_unlabeled(
        lambda triple: {"G1": 0.9, "G2": 0.2, "G3": 0.1}[triple[2]],
        [("D1", "associated_with", "G1")],
        [
            ("D1", "associated_with", "G2"),
            ("D1", "associated_with", "G3"),
        ],
        strategy="hard",
    )

    assert result["candidate_prevalence"] == 1 / 3
    assert result["label_semantics"]["0"] == "unlabeled"  # type: ignore[index]
    assert "prevalence" in str(result["candidate_prevalence_warning"])
    assert brier_score([1, 0], [0.8, 0.2]) == pytest.approx(0.04)
    assert expected_calibration_error([1, 0], [0.8, 0.2], bins=2) == pytest.approx(0.2)
