"""Ablation 3: homogeneous, relational, and heterogeneous-attention models."""

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

GRAPH_SAGE_ARM = "homogeneous GraphSAGE"
RGCN_ARM = "relational R-GCN"
HGT_ARM = "heterogeneous-attention HGT"


def run_ablation3(
    graphsage_evaluator: SeedEvaluator,
    rgcn_evaluator: SeedEvaluator,
    hgt_evaluator: SeedEvaluator,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: PathLike | None = None,
) -> AblationReport:
    """Run homogeneous vs relational vs heterogeneous attention comparison."""

    graphsage = run_arm(
        GRAPH_SAGE_ARM,
        graphsage_evaluator,
        seeds=seeds,
        config={"model": "graphsage_homogeneous"},
    )
    rgcn = run_arm(RGCN_ARM, rgcn_evaluator, seeds=seeds, config={"model": "rgcn"})
    hgt = run_arm(HGT_ARM, hgt_evaluator, seeds=seeds, config={"model": "hgt"})
    best = max((graphsage, rgcn, hgt), key=metric_mean)
    narrative = f"{best.name} has the highest mean AUPRC; this quantifies whether edge-type modeling and attention help beyond simpler structure baselines."
    report = finalize_report(
        name="ablation3_homo_rel_hetero",
        arms=[graphsage, rgcn, hgt],
        hgt_reference_arm=HGT_ARM,
        narrative=narrative,
    )
    if output_dir is not None:
        write_report(report, output_dir)
    return report
