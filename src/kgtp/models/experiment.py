"""Configuration-driven multi-seed GNN experiment runner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from kgtp.data.common import PathLike
from kgtp.eval.metrics import Triple
from kgtp.eval.runner import paired_significance, summarize_seed_metrics
from kgtp.hetero.splits import SplitBundle
from kgtp.models.train import (
    EvaluationReference,
    ModelName,
    TrainingConfig,
    evaluate_split_detailed,
    train_one_seed,
)

DEFAULT_MODELS: tuple[ModelName, ...] = (
    "hgt",
    "graphsage",
    "graphsage_homogeneous",
    "rgcn",
)


@dataclass(frozen=True)
class GNNExperimentConfig:
    """Shared model grid, seeds, and training configuration."""

    seeds: tuple[int, ...]
    models: tuple[ModelName, ...]
    training: TrainingConfig
    schema_version: int = 1


def load_experiment_config(
    path: PathLike,
    *,
    max_epochs_override: int | None = None,
) -> GNNExperimentConfig:
    """Load and validate the sample GNN experiment YAML."""

    payload = cast(
        dict[str, Any],
        yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )
    seeds = tuple(int(seed) for seed in payload["seeds"])
    if len(seeds) < 5:
        msg = "Phase 5 GNN experiments require at least five fixed seeds"
        raise ValueError(msg)
    models = cast(
        tuple[ModelName, ...],
        tuple(str(name) for name in payload["models"]),
    )
    invalid = set(models) - set(DEFAULT_MODELS)
    if invalid:
        msg = f"Unsupported GNN experiment models: {sorted(invalid)}"
        raise ValueError(msg)
    training = cast(dict[str, Any], payload["training"])
    task_weights = cast(dict[str, Any], payload["task_weights"])
    max_epochs = (
        max_epochs_override
        if max_epochs_override is not None
        else int(training["max_epochs"])
    )
    config = TrainingConfig(
        hidden_channels=int(training["hidden_channels"]),
        num_layers=int(training["num_layers"]),
        num_heads=int(training["num_heads"]),
        decoder_name=cast(Any, str(training["decoder_name"])),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_epochs=max_epochs,
        patience=min(int(training["patience"]), max_epochs),
        dropout=float(training["dropout"]),
        residual=bool(training["residual"]),
        normalization=bool(training["normalization"]),
        device=str(training["device"]),
        disease_gene_weight=float(task_weights["disease_gene"]),
        drug_gene_weight=float(task_weights["drug_gene"]),
        gene_pathway_weight=float(task_weights["gene_pathway"]),
    )
    return GNNExperimentConfig(
        seeds=seeds,
        models=models,
        training=config,
        schema_version=int(payload.get("schema_version", 1)),
    )


def run_gnn_experiment(
    bundle: SplitBundle,
    reference: EvaluationReference,
    *,
    experiment: GNNExperimentConfig,
    artifact_metadata: dict[str, str],
    sampled_unlabeled: Mapping[str, Sequence[Triple]],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    popularity_auprc: float,
    models_dir: PathLike,
    metrics_dir: PathLike,
    comparisons_dir: PathLike,
    report_path: PathLike,
) -> dict[str, Any]:
    """Train four GNN families on identical splits and persist all results."""

    model_root = Path(models_dir)
    metric_root = Path(metrics_dir)
    comparison_root = Path(comparisons_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    model_seed_metrics: dict[str, dict[str, dict[str, float]]] = {}
    model_payloads: dict[str, dict[str, Any]] = {}

    sampled = {
        name: list(triples) for name, triples in sorted(sampled_unlabeled.items())
    }
    for model_name in experiment.models:
        seed_metrics: dict[str, dict[str, float]] = {}
        per_seed_paths: dict[str, str] = {}
        for seed in experiment.seeds:
            config = replace(experiment.training, model_name=model_name)
            output_dir = model_root / model_name / f"seed_{seed}"
            result = train_one_seed(
                bundle.train_data,
                bundle.val_data,
                bundle.test_data,
                reference,
                seed=seed,
                config=config,
                output_dir=output_dir,
                artifact_metadata=artifact_metadata,
            )
            detailed = evaluate_split_detailed(
                result.model,
                bundle.test_data,
                reference,
                edge_types=config.edge_types,
                sampled_unlabeled=sampled,
            )
            flat = flatten_numeric_metrics(detailed)
            primary_auprc = flat[
                "tasks.disease__associated_with__gene.full_candidate.AUPRC"
            ]
            flat["primary.lift_over_popularity_AUPRC_ratio"] = (
                primary_auprc / popularity_auprc
                if popularity_auprc > 0
                else float("nan")
            )
            flat["primary.lift_over_popularity_AUPRC_delta"] = (
                primary_auprc - popularity_auprc
            )
            seed_metrics[str(seed)] = flat
            seed_path = metric_root / model_name / f"seed_{seed}.json"
            _write_json(
                seed_path,
                {
                    "model": model_name,
                    "seed": seed,
                    "artifact_metadata": artifact_metadata,
                    "training_config": config.__dict__,
                    "best_epoch": result.best_epoch,
                    "metrics": detailed,
                    "flat_metrics": flat,
                },
            )
            per_seed_paths[str(seed)] = str(seed_path)
        summary = summarize_seed_metrics(seed_metrics)
        model_seed_metrics[model_name] = seed_metrics
        model_payload = {
            "model": model_name,
            "seeds": list(experiment.seeds),
            "artifact_metadata": artifact_metadata,
            "training_config": replace(
                experiment.training,
                model_name=model_name,
            ).__dict__,
            "seed_metric_files": per_seed_paths,
            "seed_results": seed_metrics,
            "summary": summary,
        }
        _write_json(metric_root / model_name / "summary.json", model_payload)
        model_payloads[model_name] = model_payload

    primary_metric = "tasks.disease__associated_with__gene.full_candidate.AUPRC"
    hgt_reference = {
        int(seed): {primary_metric: values[primary_metric]}
        for seed, values in model_seed_metrics["hgt"].items()
    }
    comparisons = {
        model_name: (
            {
                "status": "reference",
                "metric": primary_metric,
                "warning": "Five-seed sample comparison; not scientific evidence.",
            }
            if model_name == "hgt"
            else paired_significance(
                seed_metrics,
                hgt_reference,
                metric=primary_metric,
            )
        )
        for model_name, seed_metrics in model_seed_metrics.items()
    }
    comparison_payload = {
        "scope": "sample pipeline validation; not a scientific benchmark",
        "primary_metric": primary_metric,
        "identical_split_hash": artifact_metadata["split_hash"],
        "seeds": list(experiment.seeds),
        "models": model_payloads,
        "comparisons_vs_hgt": comparisons,
        "popularity_reference": {
            "AUPRC": popularity_auprc,
            "seed_dependence": "none; train-graph popularity is deterministic",
        },
        "baseline_references": {
            name: {
                "AUPRC": metrics.get("AUPRC"),
                "AUROC": metrics.get("AUROC"),
                "scope": (
                    "single-seed Phase 4 reference; descriptive only and not used "
                    "for paired significance testing"
                ),
            }
            for name, metrics in sorted(baseline_metrics.items())
        },
        "limitations": [
            "Only five seeds are available.",
            "The sample graph is small and transductive.",
            "AUPRC values are comparable only within the same candidate protocol.",
            "Most Phase 4 baselines are single-seed descriptive references.",
            "No multiple-comparison correction is claimed for this small model grid.",
        ],
    }
    comparison_path = comparison_root / "gnn_comparison.json"
    _write_json(comparison_path, comparison_payload)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_benchmark_markdown(comparison_payload),
        encoding="utf-8",
    )
    return comparison_payload


def load_sampled_unlabeled(
    sampling_dir: PathLike,
    *,
    relation: str = "associated_with",
) -> dict[str, list[Triple]]:
    """Load persisted Phase 4 unlabeled pairs for identical GNN evaluation."""

    root = Path(sampling_dir)
    return {
        name: [
            (str(row.source_id), relation, str(row.target_id))
            for row in pd.read_parquet(root / f"{name}.parquet").itertuples(index=False)
        ]
        for name in ("random", "degree_matched", "hard")
    }


def flatten_numeric_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten numeric leaves for seed summaries and paired comparisons."""

    output: dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_numeric_metrics(value, prefix=path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            output[path] = float(value)
    return output


def render_benchmark_markdown(payload: Mapping[str, Any]) -> str:
    """Render the sample benchmark table exclusively from saved metrics."""

    models = cast(Mapping[str, Mapping[str, Any]], payload["models"])
    primary = str(payload["primary_metric"])
    lines = [
        "# Sample GNN benchmark",
        "",
        "This is deterministic software-validation output on a small redistributable "
        "sample. It is not a scientific biomedical benchmark.",
        "",
        f"- Seeds: `{payload['seeds']}`",
        f"- Split hash: `{payload['identical_split_hash']}`",
        "- Primary task: `disease -> associated_with -> gene`",
        "- Primary protocol: full-candidate ranking over all eligible genes",
        "",
        "| Model | AUPRC mean | std | 95% CI half-width | MRR mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    mrr_metric = "tasks.disease__associated_with__gene.full_candidate.filtered.MRR"
    for name in sorted(models):
        summary = cast(Mapping[str, Mapping[str, float]], models[name]["summary"])
        auprc = summary[primary]
        mrr = summary[mrr_metric]
        lines.append(
            f"| {name} | {auprc['mean']:.4f} | {auprc['std']:.4f} | "
            f"{auprc['ci95']:.4f} | {mrr['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Five seeds support variability reporting, not strong significance claims.",
            "- Sampled-unlabeled and full-candidate AUPRC use different prevalence and "
            "must not be compared as though they were the same estimand.",
            "- Unobserved pairs are unlabeled, not confirmed biological negatives.",
            "- Full-batch training is used only because the sample graph is small.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
