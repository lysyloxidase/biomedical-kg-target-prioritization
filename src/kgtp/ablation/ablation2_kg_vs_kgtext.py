"""Ablation 2: KG-only features vs KG plus text embeddings."""

from __future__ import annotations

from collections.abc import Sequence

from kgtp.ablation.common import (
    DEFAULT_SEEDS,
    AblationReport,
    SeedEvaluator,
    finalize_report,
    metric_mean,
    run_arm,
    write_report,
)
from kgtp.data.common import PathLike

KG_ONLY_ARM = "KG-only (HGT structural)"
KG_TEXT_ARM = "KG+text (HGT + PubMedBERT)"


def run_ablation2(
    kg_only_evaluator: SeedEvaluator,
    kg_text_evaluator: SeedEvaluator,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: PathLike | None = None,
) -> AblationReport:
    """Run KG-only vs KG+text and quantify text value over graph structure."""

    kg_only = run_arm(
        KG_ONLY_ARM, kg_only_evaluator, seeds=seeds, config={"features": "structural"}
    )
    kg_text = run_arm(
        KG_TEXT_ARM, kg_text_evaluator, seeds=seeds, config={"features": "text"}
    )
    delta = metric_mean(kg_text) - metric_mean(kg_only)
    narrative = (
        f"Text embeddings add {delta:.4f} mean AUPRC over KG-only HGT."
        if delta > 0
        else f"KG+text does not improve over KG-only HGT (delta AUPRC {delta:.4f}); text co-occurrence is not adding signal here."
    )
    report = finalize_report(
        name="ablation2_kg_vs_kgtext",
        arms=[kg_only, kg_text],
        hgt_reference_arm=KG_ONLY_ARM,
        narrative=narrative,
    )
    if output_dir is not None:
        write_report(report, output_dir)
    return report
