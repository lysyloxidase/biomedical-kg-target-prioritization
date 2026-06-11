from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from kgtp.ablation.ablation1_nokg_vs_kg import KG_ARM, NO_KG_ARM, run_ablation1
from kgtp.ablation.ablation2_kg_vs_kgtext import KG_TEXT_ARM, run_ablation2
from kgtp.ablation.ablation3_homo_rel_hetero import HGT_ARM, run_ablation3
from kgtp.ablation.ablation4_design_knobs import (
    FEATURE_CHOICES,
    NEGATIVE_STRATEGIES,
    NUM_LAYERS,
    design_grid_configs,
    run_ablation4,
)
from kgtp.ablation.common import read_report, write_report
from kgtp.ablation.tables import (
    ablation3_markdown,
    ablation4_markdown,
    ablation12_markdown,
    write_ablation_tables,
)

SEEDS = (13, 17, 19, 23, 29)


def _metrics(
    base_auprc: float, seed: int, *, auroc: float | None = None
) -> dict[str, float]:
    jitter = (seed % 7) * 0.001
    return {
        "AUROC": (base_auprc + 0.2 if auroc is None else auroc) + jitter,
        "AUPRC": base_auprc + jitter,
        "filtered_Hits@10": base_auprc + 0.1 + jitter,
        "filtered_MRR": base_auprc - 0.05 + jitter,
    }


def test_ablation1_reports_honest_no_kg_win_and_significance(tmp_path) -> None:
    report = run_ablation1(
        lambda seed: _metrics(0.42, seed),
        lambda seed: _metrics(0.38, seed),
        seeds=SEEDS,
        output_dir=tmp_path,
    )

    assert report.hgt_reference_arm == KG_ARM
    assert "does not beat" in report.narrative
    assert len(report.arms[0].seed_results) == 5
    assert report.arms[0].significance_vs_hgt is not None
    assert report.arms[1].significance_vs_hgt["status"] == "reference"  # type: ignore[index]
    assert (tmp_path / "ablation1_nokg_vs_kg.json").exists()

    loaded = read_report(tmp_path / "ablation1_nokg_vs_kg.json")
    assert (
        loaded.arms[0].summary["AUPRC"]["mean"]
        == report.arms[0].summary["AUPRC"]["mean"]
    )


def test_ablation2_and_3_quantify_text_and_heterogeneity() -> None:
    ablation2 = run_ablation2(
        lambda seed: _metrics(0.50, seed),
        lambda seed: _metrics(0.56, seed),
        seeds=SEEDS,
    )
    ablation3 = run_ablation3(
        lambda seed: _metrics(0.47, seed),
        lambda seed: _metrics(0.51, seed),
        lambda seed: _metrics(0.55, seed),
        seeds=SEEDS,
    )

    assert KG_TEXT_ARM in {arm.name for arm in ablation2.arms}
    assert "Text embeddings add" in ablation2.narrative
    assert ablation3.hgt_reference_arm == HGT_ARM
    assert all(len(arm.seed_results) == 5 for arm in ablation3.arms)
    assert all(arm.significance_vs_hgt is not None for arm in ablation3.arms)


def test_ablation4_design_grid_is_complete_and_warns_about_random_auroc() -> None:
    def evaluator(config: Mapping[str, object], seed: int) -> dict[str, float]:
        negative = str(config["negative_sampling"])
        layers = int(cast(int, config["num_layers"]))
        features = str(config["features"])
        auroc_by_negative = {"random": 0.92, "degree_matched": 0.84, "hard": 0.77}
        feature_bonus = {
            "one-hot": 0.00,
            "structural": 0.03,
            "ESM": 0.04,
            "text": 0.05,
        }[features]
        return _metrics(
            0.35 + 0.02 * layers + feature_bonus,
            seed,
            auroc=auroc_by_negative[negative],
        )

    report = run_ablation4(evaluator, seeds=SEEDS)

    assert len(design_grid_configs()) == len(NEGATIVE_STRATEGIES) * len(
        NUM_LAYERS
    ) * len(FEATURE_CHOICES)
    assert len(report.arms) == 36
    assert "AUROC inflation warning" in report.narrative
    assert all(len(arm.seed_results) == 5 for arm in report.arms)
    assert all(arm.significance_vs_hgt is not None for arm in report.arms)


def test_ablation_tables_markdown_latex_and_narrative_are_written(tmp_path) -> None:
    ablation1 = run_ablation1(
        lambda seed: _metrics(0.42, seed),
        lambda seed: _metrics(0.45, seed),
        seeds=SEEDS,
    )
    ablation2 = run_ablation2(
        lambda seed: _metrics(0.45, seed),
        lambda seed: _metrics(0.48, seed),
        seeds=SEEDS,
    )
    ablation3 = run_ablation3(
        lambda seed: _metrics(0.40, seed),
        lambda seed: _metrics(0.43, seed),
        lambda seed: _metrics(0.45, seed),
        seeds=SEEDS,
    )
    ablation4 = run_ablation4(
        lambda config, seed: _metrics(
            0.36 + 0.01 * int(cast(int, config["num_layers"])), seed
        ),
        seeds=SEEDS,
    )

    paths = write_ablation_tables(ablation1, ablation2, ablation3, ablation4, tmp_path)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    latex = paths["latex"].read_text(encoding="utf-8")
    narrative = paths["narrative"].read_text(encoding="utf-8")

    assert NO_KG_ARM in ablation12_markdown(ablation1, ablation2)
    assert HGT_ARM in ablation3_markdown(ablation3)
    assert "neg=random" in ablation4_markdown(ablation4)
    assert "±" in markdown
    assert "$\\pm$" in latex
    assert "Honest Ablation Narrative" in narrative

    saved_path = write_report(ablation1, tmp_path / "saved")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["name"] == "ablation1_nokg_vs_kg"
