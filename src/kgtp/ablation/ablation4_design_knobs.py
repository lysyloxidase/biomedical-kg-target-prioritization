"""Ablation 4: design-knob grid over negatives, layers, and features."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from kgtp.ablation.common import (
    DEFAULT_SEEDS,
    AblationArm,
    AblationReport,
    SeedEvaluator,
    finalize_report,
    metric_mean,
    run_arm,
    write_report,
)
from kgtp.data.common import PathLike

NEGATIVE_STRATEGIES: tuple[str, ...] = ("random", "degree_matched", "hard")
NUM_LAYERS: tuple[int, ...] = (1, 2, 3)
FEATURE_CHOICES: tuple[str, ...] = ("one-hot", "structural", "ESM", "text")
BASELINE_CONFIG = {
    "negative_sampling": "random",
    "num_layers": 2,
    "features": "structural",
}
DesignConfig = Mapping[str, object]
DesignEvaluator = Callable[[DesignConfig, int], Mapping[str, object]]


def design_grid_configs() -> list[dict[str, object]]:
    """Return the complete Phase 5 design-knob grid."""

    return [
        {"negative_sampling": negative, "num_layers": layers, "features": features}
        for negative in NEGATIVE_STRATEGIES
        for layers in NUM_LAYERS
        for features in FEATURE_CHOICES
    ]


def design_arm_name(config: DesignConfig) -> str:
    """Stable table row name for a grid cell."""

    return (
        f"neg={config['negative_sampling']} | "
        f"layers={config['num_layers']} | "
        f"features={config['features']}"
    )


def run_ablation4(
    evaluator: DesignEvaluator,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: PathLike | None = None,
) -> AblationReport:
    """Run the full design-knob grid."""

    arms = [
        run_arm(
            design_arm_name(config),
            _seed_evaluator(evaluator, config),
            seeds=seeds,
            config=config,
        )
        for config in design_grid_configs()
    ]
    baseline_name = design_arm_name(BASELINE_CONFIG)
    random_mean = _best_negative_strategy_mean(arms, "random")
    hard_mean = _best_negative_strategy_mean(arms, "hard")
    narrative = (
        f"Random negatives exceed hard negatives by {random_mean - hard_mean:.4f} best-cell AUROC; this is the expected AUROC inflation warning."
        if random_mean > hard_mean
        else f"Hard negatives match or exceed random negatives by {hard_mean - random_mean:.4f} best-cell AUROC; random-negative AUROC is not inflated in this run."
    )
    report = finalize_report(
        name="ablation4_design_knobs",
        arms=arms,
        hgt_reference_arm=baseline_name,
        narrative=narrative,
    )
    if output_dir is not None:
        write_report(report, output_dir)
    return report


def _seed_evaluator(evaluator: DesignEvaluator, config: DesignConfig) -> SeedEvaluator:
    def evaluate(seed: int) -> Mapping[str, object]:
        return evaluator(config, seed)

    return evaluate


def _best_negative_strategy_mean(arms: Sequence[AblationArm], strategy: str) -> float:
    matching = [arm for arm in arms if arm.config.get("negative_sampling") == strategy]
    return max(metric_mean(arm, "AUROC") for arm in matching)
