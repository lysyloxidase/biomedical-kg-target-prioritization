"""Multi-seed evaluation harness for baselines and GNNs."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kgtp.data.common import PathLike

MetricDict = Mapping[str, float]
SeedEvaluator = Callable[[int], MetricDict]


def run_multiseed(
    model_name: str,
    evaluator: SeedEvaluator,
    *,
    seeds: Sequence[int] = (13, 17, 19, 23, 29),
    output_dir: PathLike = "reports",
    hgt_reference: Mapping[int, MetricDict] | None = None,
) -> dict[str, Any]:
    """Run a model across seeds and write ``reports/results_<model>.json``."""

    if len(seeds) < 5:
        msg = "Phase 3 requires at least five evaluation seeds"
        raise ValueError(msg)

    seed_results = {str(seed): dict(evaluator(seed)) for seed in seeds}
    summary = summarize_seed_metrics(seed_results)
    significance = paired_significance(seed_results, hgt_reference)
    payload = {
        "model": model_name,
        "primary_metric": "AUPRC",
        "auroc_note": "AUROC is reported but optimistic under class imbalance.",
        "seeds": list(seeds),
        "seed_results": seed_results,
        "summary": summary,
        "paired_significance_vs_hgt": significance,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"results_{model_name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def summarize_seed_metrics(
    seed_results: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Return mean, std, and 95% CI for every scalar metric."""

    metric_names = sorted(
        {metric for result in seed_results.values() for metric in result}
    )
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray(
            [result[metric] for result in seed_results.values()], dtype=float
        )
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) else math.nan
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else math.nan,
            "std": std,
            "ci95": float(ci95),
        }
    return summary


def paired_significance(
    seed_results: Mapping[str, Mapping[str, float]],
    hgt_reference: Mapping[int, Mapping[str, float]] | None,
    *,
    metric: str = "AUPRC",
) -> dict[str, object]:
    """Return paired parametric and resampling comparisons."""

    if hgt_reference is None:
        return {
            "test": "paired_t_test",
            "metric": metric,
            "status": "pending_phase_4_hgt_reference",
        }
    common_seeds = sorted(
        seed
        for seed in (int(seed_text) for seed_text in seed_results)
        if seed in hgt_reference
    )
    if len(common_seeds) < 2:
        return {
            "test": "paired_t_test",
            "metric": metric,
            "status": "insufficient_pairs",
        }

    baseline_values = np.asarray(
        [seed_results[str(seed)][metric] for seed in common_seeds], dtype=float
    )
    hgt_values = np.asarray(
        [hgt_reference[seed][metric] for seed in common_seeds], dtype=float
    )
    differences = baseline_values - hgt_values
    statistic = _paired_t_statistic(differences)
    p_value = _paired_p_value(baseline_values, hgt_values, statistic)
    bootstrap_low, bootstrap_high = _paired_bootstrap_ci(differences)
    return {
        "test": "paired_t_test",
        "metric": metric,
        "n": len(common_seeds),
        "statistic": statistic,
        "p_value": p_value,
        "assumptions": (
            "The paired t-test assumes approximately normal seed-wise differences; "
            "five seeds provide weak evidence and do not establish significance."
        ),
        "mean_difference": float(differences.mean()),
        "paired_bootstrap_95_ci": [bootstrap_low, bootstrap_high],
        "paired_sign_flip_p_value": _paired_sign_flip_p_value(differences),
        "interpretation_warning": (
            "Treat all comparisons as exploratory because n=5 and the sample graph "
            "is not a scientific benchmark."
        ),
    }


def _paired_t_statistic(differences: np.ndarray) -> float:
    if len(differences) < 2:
        return math.nan
    std = float(differences.std(ddof=1))
    if std == 0.0:
        return math.inf if float(differences.mean()) != 0.0 else 0.0
    return float(differences.mean() / (std / math.sqrt(len(differences))))


def _paired_p_value(first: np.ndarray, second: np.ndarray, statistic: float) -> float:
    try:
        stats = importlib.import_module("scipy.stats")
    except ModuleNotFoundError:
        return float(math.erfc(abs(statistic) / math.sqrt(2.0)))
    result = stats.ttest_rel(first, second)
    return float(result.pvalue)


def _paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 13,
) -> tuple[float, float]:
    if len(differences) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_sign_flip_p_value(differences: np.ndarray) -> float:
    """Return an exact two-sided paired sign-flip permutation p-value."""

    count = len(differences)
    if count == 0:
        return math.nan
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 2**count
    for mask in range(total):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(count)]
        )
        permuted = abs(float((differences * signs).mean()))
        if permuted >= observed - 1e-12:
            extreme += 1
    return float(extreme / total)
