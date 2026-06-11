"""Ablation 1: no-KG feature baseline vs KG message passing."""

from __future__ import annotations

from collections.abc import Sequence

from kgtp.ablation.common import (
    DEFAULT_SEEDS,
    AblationReport,
    SeedEvaluator,
    finalize_report,
    graph_help_narrative,
    run_arm,
    write_report,
)
from kgtp.data.common import PathLike

NO_KG_ARM = "no-KG (LR/MLP features)"
KG_ARM = "KG (HGT structural)"


def run_ablation1(
    no_kg_evaluator: SeedEvaluator,
    kg_evaluator: SeedEvaluator,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: PathLike | None = None,
) -> AblationReport:
    """Run no-KG vs KG with identical splits/features."""

    no_kg = run_arm(
        NO_KG_ARM, no_kg_evaluator, seeds=seeds, config={"message_passing": False}
    )
    kg = run_arm(KG_ARM, kg_evaluator, seeds=seeds, config={"message_passing": True})
    report = finalize_report(
        name="ablation1_nokg_vs_kg",
        arms=[no_kg, kg],
        hgt_reference_arm=KG_ARM,
        narrative=graph_help_narrative(no_kg, kg),
    )
    if output_dir is not None:
        write_report(report, output_dir)
    return report
