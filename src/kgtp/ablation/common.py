"""Shared ablation reporting utilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kgtp.data.common import PathLike
from kgtp.eval.runner import paired_significance, summarize_seed_metrics

DEFAULT_SEEDS: tuple[int, ...] = (13, 17, 19, 23, 29)
PRIMARY_METRIC = "AUPRC"
TABLE_METRICS: tuple[str, ...] = ("AUROC", "AUPRC", "Hits@10", "MRR")
SeedEvaluator = Callable[[int], Mapping[str, Any]]


@dataclass
class AblationArm:
    """One row in an ablation table."""

    name: str
    seed_results: dict[str, dict[str, float]]
    config: dict[str, object] = field(default_factory=dict)
    summary: dict[str, dict[str, float]] = field(init=False)
    significance_vs_hgt: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.seed_results = {
            str(seed): standardize_metrics(metrics)
            for seed, metrics in self.seed_results.items()
        }
        require_minimum_seeds(self.seed_results)
        self.summary = summarize_seed_metrics(self.seed_results)

    def to_dict(self) -> dict[str, object]:
        """Serialize this arm to a JSON-compatible mapping."""

        return {
            "name": self.name,
            "config": self.config,
            "seed_results": self.seed_results,
            "summary": self.summary,
            "significance_vs_hgt": self.significance_vs_hgt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AblationArm:
        """Rebuild an arm from saved metrics."""

        arm = cls(
            name=str(payload["name"]),
            seed_results={
                str(seed): dict(metrics)
                for seed, metrics in payload["seed_results"].items()
            },
            config=dict(payload.get("config", {})),
        )
        significance = payload.get("significance_vs_hgt")
        arm.significance_vs_hgt = (
            dict(significance) if isinstance(significance, dict) else None
        )
        return arm


@dataclass
class AblationReport:
    """Full ablation report with rows, significance, and narrative."""

    name: str
    arms: list[AblationArm]
    hgt_reference_arm: str
    narrative: str
    primary_metric: str = PRIMARY_METRIC

    def to_dict(self) -> dict[str, object]:
        """Serialize this report to JSON."""

        return {
            "name": self.name,
            "primary_metric": self.primary_metric,
            "hgt_reference_arm": self.hgt_reference_arm,
            "narrative": self.narrative,
            "arms": [arm.to_dict() for arm in self.arms],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AblationReport:
        """Rebuild a report from saved JSON."""

        return cls(
            name=str(payload["name"]),
            primary_metric=str(payload.get("primary_metric", PRIMARY_METRIC)),
            hgt_reference_arm=str(payload["hgt_reference_arm"]),
            narrative=str(payload["narrative"]),
            arms=[AblationArm.from_dict(arm) for arm in payload["arms"]],
        )


def run_arm(
    name: str,
    evaluator: SeedEvaluator,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config: Mapping[str, object] | None = None,
) -> AblationArm:
    """Run one ablation arm across seeds."""

    if len(seeds) < 5:
        msg = "Ablations require at least five seeds per cell"
        raise ValueError(msg)
    seed_results = {str(seed): dict(evaluator(seed)) for seed in seeds}
    return AblationArm(name=name, seed_results=seed_results, config=dict(config or {}))


def finalize_report(
    *,
    name: str,
    arms: Sequence[AblationArm],
    hgt_reference_arm: str,
    narrative: str,
) -> AblationReport:
    """Attach paired significance against the HGT reference arm."""

    hgt_arm = next((arm for arm in arms if arm.name == hgt_reference_arm), None)
    if hgt_arm is None:
        msg = f"Missing HGT reference arm: {hgt_reference_arm}"
        raise ValueError(msg)
    hgt_reference = {
        int(seed): metrics for seed, metrics in hgt_arm.seed_results.items()
    }
    finalized = list(arms)
    for arm in finalized:
        if arm.name == hgt_reference_arm:
            arm.significance_vs_hgt = {
                "test": "paired_t_test",
                "metric": PRIMARY_METRIC,
                "status": "reference",
            }
        else:
            arm.significance_vs_hgt = paired_significance(
                arm.seed_results,
                hgt_reference,
                metric=PRIMARY_METRIC,
            )
    return AblationReport(
        name=name,
        arms=finalized,
        hgt_reference_arm=hgt_reference_arm,
        narrative=narrative,
    )


def standardize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Normalize Phase 3/4 metric names to ablation-table columns."""

    filtered = metrics.get("filtered")
    filtered_map = filtered if isinstance(filtered, Mapping) else {}
    out: dict[str, float] = {}
    for metric in ("AUROC", "AUPRC"):
        if metric in metrics:
            out[metric] = float(metrics[metric])
    out["Hits@10"] = float(
        metrics.get(
            "Hits@10", metrics.get("filtered_Hits@10", filtered_map.get("Hits@10", 0.0))
        )
    )
    out["MRR"] = float(
        metrics.get("MRR", metrics.get("filtered_MRR", filtered_map.get("MRR", 0.0)))
    )
    return out


def require_minimum_seeds(seed_results: Mapping[str, Mapping[str, float]]) -> None:
    """Raise unless every arm has the Phase 5 minimum seed count."""

    if len(seed_results) < 5:
        msg = "Every ablation cell must contain metrics for at least five seeds"
        raise ValueError(msg)


def metric_mean(arm: AblationArm, metric: str = PRIMARY_METRIC) -> float:
    """Return a metric mean from an arm summary."""

    return arm.summary[metric]["mean"]


def graph_help_narrative(no_kg_arm: AblationArm, kg_arm: AblationArm) -> str:
    """Write the honest no-KG-vs-KG narrative required by Phase 5."""

    no_kg = metric_mean(no_kg_arm)
    kg = metric_mean(kg_arm)
    delta = kg - no_kg
    if delta > 0:
        return (
            f"HGT improves mean AUPRC by {delta:.4f} over the no-KG feature baseline; "
            "under these splits, message passing adds measurable signal."
        )
    return (
        f"HGT does not beat the no-KG feature baseline (delta AUPRC {delta:.4f}); "
        "the honest finding is that graph propagation adds little under this sparsity."
    )


def write_report(report: AblationReport, output_dir: PathLike) -> Path:
    """Persist a report so tables can be reproduced from saved seed metrics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{report.name}.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def read_report(path: PathLike) -> AblationReport:
    """Load a saved ablation report."""

    return AblationReport.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
