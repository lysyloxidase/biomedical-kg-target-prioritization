"""Deterministic end-to-end pipeline for the redistributable OA sample."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from kgtp.baselines.runner import run_baseline_suite
from kgtp.data.build_graph import (
    assemble_canonical_edges,
    assert_canonical_edge_types,
    write_graph_tables,
)
from kgtp.data.common import (
    ensure_columns,
    normalize_ensembl_gene_id,
    read_table,
    stable_drop_duplicates,
    write_table,
)
from kgtp.eval.metrics import Triple
from kgtp.hetero.auxiliary_splits import (
    attach_auxiliary_supervision,
    load_auxiliary_splits,
    split_auxiliary_relations,
    write_auxiliary_splits,
)
from kgtp.hetero.build_heterodata import build_heterodata
from kgtp.hetero.feature_transformers import TrainGraphFeatureTransformer
from kgtp.hetero.split_protocol import (
    build_split_bundle,
    hash_graph,
    load_target_split,
    split_target_relation,
    validate_target_split,
    write_target_split,
)
from kgtp.hetero.splits import load_splits, save_splits
from kgtp.kg.statistics import GraphStatistics
from kgtp.models.experiment import (
    flatten_numeric_metrics,
    load_experiment_config,
    load_sampled_unlabeled,
    run_gnn_experiment,
)
from kgtp.models.train import (
    EvaluationReference,
    evaluate_split_detailed,
    load_training_checkpoint,
)
from kgtp.run_manifest import (
    complete_run_manifest,
    default_command,
    fail_run_manifest,
    hashes_for_files,
    sha256_file,
    start_run_manifest,
    update_run_manifest,
    validate_run_compatibility,
)

SEED = 13
DISEASE_GENE_EDGE = ("disease", "associated_with", "gene")
REVERSE_DISEASE_GENE_EDGE = ("gene", "rev_associated_with", "disease")
SAMPLE_TABLES = (
    "nodes.parquet",
    "disease_gene.parquet",
    "gene_gene.parquet",
    "gene_pathway.parquet",
    "drug_gene.parquet",
    "gene_go.parquet",
    "pathway_pathway.parquet",
)


@dataclass(frozen=True)
class SamplePaths:
    """Filesystem contract for the small sample pipeline."""

    sample_dir: Path = Path("data/sample")
    root: Path = Path("artifacts/sample")

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def graph(self) -> Path:
        return self.root / "graph"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def train_graph(self) -> Path:
        return self.root / "train_message_graph"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def report(self) -> Path:
        return self.root / "report"


DEFAULT_PATHS = SamplePaths()


def prepare_sample(paths: SamplePaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Validate the checked-in snapshot and record its source manifest."""

    manifest_path = _require_file(paths.sample_dir / "manifest.json")
    source_manifest = _read_json(manifest_path)
    files = cast(dict[str, dict[str, Any]], source_manifest.get("files", {}))
    if set(files) != set(SAMPLE_TABLES):
        msg = "Sample manifest does not list exactly the required Parquet tables"
        raise ValueError(msg)

    input_paths = [manifest_path]
    for filename in SAMPLE_TABLES:
        path = _require_file(paths.sample_dir / filename)
        input_paths.append(path)
        expected = files[filename]
        if _sha256(path) != expected.get("sha256"):
            msg = f"Checksum mismatch for {path}"
            raise ValueError(msg)
        frame = read_table(path)
        if frame.empty:
            msg = f"Required sample table is empty: {path}"
            raise ValueError(msg)
        if len(frame) != int(expected.get("rows", -1)):
            msg = f"Row-count mismatch for {path}"
            raise ValueError(msg)

    output = paths.manifests / "source_manifest.json"
    _write_json(output, source_manifest)
    payload = _write_stage_manifest(
        paths,
        "prepare-sample",
        inputs=input_paths,
        outputs=[output],
        metadata={
            "dataset": source_manifest.get("dataset"),
            "retrieval_date": source_manifest.get("retrieval_date"),
            "validated_tables": len(SAMPLE_TABLES),
        },
    )
    _log_stage(payload)
    return payload


def normalize_sample(paths: SamplePaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Normalize identifiers and deterministically rewrite all sample tables."""

    _require_file(paths.manifests / "prepare-sample.json")
    tables = {
        filename: read_table(_require_file(paths.sample_dir / filename))
        for filename in SAMPLE_TABLES
    }
    normalized = _normalize_tables(tables)
    outputs: list[Path] = []
    for filename in SAMPLE_TABLES:
        output = paths.normalized / filename
        write_table(normalized[filename], output)
        outputs.append(output)

    payload = _write_stage_manifest(
        paths,
        "normalize",
        inputs=[paths.sample_dir / filename for filename in SAMPLE_TABLES],
        outputs=outputs,
        metadata={
            "rows": {filename: len(normalized[filename]) for filename in SAMPLE_TABLES}
        },
    )
    _log_stage(payload)
    return payload


def assemble_sample(paths: SamplePaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Assemble canonical graph tables without requiring Neo4j."""

    _require_file(paths.manifests / "normalize.json")
    normalized = {
        filename: read_table(_require_file(paths.normalized / filename))
        for filename in SAMPLE_TABLES
    }
    edge_tables = assemble_canonical_edges(
        disease_gene=normalized["disease_gene.parquet"],
        gene_gene=normalized["gene_gene.parquet"],
        gene_pathway=normalized["gene_pathway.parquet"],
        drug_gene=normalized["drug_gene.parquet"],
        gene_go=normalized["gene_go.parquet"],
        pathway_pathway=normalized["pathway_pathway.parquet"],
    )
    assert_canonical_edge_types(edge_tables)
    nodes = normalized["nodes.parquet"]
    _validate_graph_endpoints(nodes, edge_tables)
    write_graph_tables(edge_tables, nodes, paths.graph)

    statistics = GraphStatistics.from_tables(nodes, edge_tables)
    stats_payload = {
        "node_counts": statistics.node_counts,
        "edge_counts": statistics.edge_counts,
        "density": statistics.density,
        "mean_degree": statistics.mean_degree,
        "positive_disease_gene_links": statistics.positive_disease_gene_links,
    }
    stats_json = paths.graph / "statistics.json"
    stats_parquet = paths.graph / "statistics.parquet"
    _write_json(stats_json, stats_payload)
    write_table(statistics.to_frame(), stats_parquet)
    outputs = [
        paths.graph / "nodes.parquet",
        *[paths.graph / "edges" / f"{name}.parquet" for name in sorted(edge_tables)],
        stats_json,
        stats_parquet,
    ]
    payload = _write_stage_manifest(
        paths,
        "assemble",
        inputs=[paths.normalized / filename for filename in SAMPLE_TABLES],
        outputs=outputs,
        metadata=stats_payload,
    )
    _log_stage(payload)
    return payload


def split_sample(
    paths: SamplePaths = DEFAULT_PATHS,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Split target edges before graph-derived feature construction."""

    _require_file(paths.manifests / "assemble.json")
    nodes, edge_tables = _read_graph_tables(paths.graph)
    split = split_target_relation(
        nodes,
        edge_tables,
        seed=seed,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.3,
    )
    auxiliary_splits, train_edge_tables = split_auxiliary_relations(
        nodes,
        edge_tables,
        seed=seed,
    )
    train_edge_tables["disease_gene"] = split.message_edges
    split.metadata["train_message_graph_hash"] = hash_graph(nodes, train_edge_tables)
    split.metadata["relation_policy"].update(
        {
            "drug_gene": "auxiliary task partitioned before feature fitting",
            "gene_pathway": "auxiliary task partitioned before feature fitting",
        }
    )
    split.metadata["auxiliary_tasks"] = {
        name: task.metadata for name, task in sorted(auxiliary_splits.items())
    }
    dataset_manifest = paths.manifests / "dataset.json"
    source_manifest = _read_json(
        _require_file(paths.manifests / "source_manifest.json")
    )
    _write_json(
        dataset_manifest,
        {
            "schema_version": 1,
            "dataset_id": str(source_manifest["dataset"]),
            "dataset_version": str(source_manifest.get("retrieval_date", "unknown")),
            "full_reference_graph_hash": split.metadata["full_reference_graph_hash"],
            "train_message_graph_hash": split.metadata["train_message_graph_hash"],
            "node_index_map_hash": split.metadata["node_index_map_hash"],
            "source_manifest_sha256": _sha256(paths.manifests / "source_manifest.json"),
        },
    )
    validate_target_split(
        split,
        nodes=nodes,
        edge_tables=edge_tables,
        train_edge_tables=train_edge_tables,
    )
    split_outputs = [
        *write_target_split(split, paths.splits),
        *write_auxiliary_splits(auxiliary_splits, paths.splits),
        dataset_manifest,
    ]
    validation_report = paths.splits / "leakage_validation.json"
    _write_json(
        validation_report,
        {
            "status": "passed",
            "target_relation": list(DISEASE_GENE_EDGE),
            "checks": [
                "reference registry matches canonical target edges",
                "positive partitions are pairwise disjoint",
                "positive partitions reconstruct the reference registry",
                "train supervision is disjoint from message edges",
                "validation and test positives are absent from message edges",
                "unlabeled supervision excludes known positives",
                "graph, partition, and node-index hashes match",
            ],
            "full_reference_graph_hash": split.metadata["full_reference_graph_hash"],
            "train_message_graph_hash": split.metadata["train_message_graph_hash"],
            "node_index_map_hash": split.metadata["node_index_map_hash"],
        },
    )
    split_outputs.append(validation_report)
    write_graph_tables(train_edge_tables, nodes, paths.train_graph)
    train_graph_outputs = [
        paths.train_graph / "nodes.parquet",
        *[
            paths.train_graph / "edges" / f"{name}.parquet"
            for name in sorted(train_edge_tables)
        ],
    ]
    payload = _write_stage_manifest(
        paths,
        "split",
        inputs=[
            paths.graph / "nodes.parquet",
            *sorted((paths.graph / "edges").glob("*.parquet")),
        ],
        outputs=[*split_outputs, *train_graph_outputs],
        metadata=split.metadata,
    )
    _log_stage(payload)
    return payload


def build_sample_features(paths: SamplePaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Fit graph features on the train message graph and build PyG views."""

    _require_file(paths.manifests / "split.json")
    reference_nodes, reference_edges = _read_graph_tables(paths.graph)
    train_nodes, train_edges = _read_graph_tables(paths.train_graph)
    split = load_target_split(paths.splits)
    validate_target_split(
        split,
        nodes=reference_nodes,
        edge_tables=reference_edges,
        train_edge_tables=train_edges,
    )
    train_graph_hash = hash_graph(train_nodes, train_edges)
    if train_graph_hash != split.metadata["train_message_graph_hash"]:
        msg = "Feature-fit graph does not match the split train-graph hash"
        raise ValueError(msg)

    transformer = TrainGraphFeatureTransformer().fit(
        train_nodes,
        train_edges,
        graph_hash=train_graph_hash,
    )
    precomputed_features = transformer.transform(train_nodes)
    message_data = build_heterodata(
        train_nodes,
        train_edges,
        output_dir=paths.features,
        gene_feature_mode="none",
        precomputed_features=precomputed_features,
    )
    if not message_data.node_types or DISEASE_GENE_EDGE not in message_data.edge_types:
        msg = "HeteroData is missing required node or disease-gene edge types"
        raise ValueError(msg)
    bundle = build_split_bundle(
        message_data,
        split,
        edge_type=DISEASE_GENE_EDGE,
        reverse_edge_type=REVERSE_DISEASE_GENE_EDGE,
    )
    bundle = attach_auxiliary_supervision(
        bundle,
        load_auxiliary_splits(paths.splits),
    )
    save_splits(bundle, paths.splits, write_metadata=False)
    transformer_path = paths.features / "feature_transformer.json"
    _write_json(transformer_path, transformer.to_dict())
    feature_shapes = {
        node_type: list(message_data[node_type].x.shape)
        for node_type in message_data.node_types
    }
    payload = _write_stage_manifest(
        paths,
        "features",
        inputs=[
            paths.train_graph / "nodes.parquet",
            *sorted((paths.train_graph / "edges").glob("*.parquet")),
            paths.splits / "split_metadata.json",
            paths.splits / "supervision" / "train.parquet",
            paths.splits / "supervision" / "validation.parquet",
            paths.splits / "supervision" / "test.parquet",
        ],
        outputs=[
            paths.features / "heterodata.pt",
            paths.features / "node_index_maps.json",
            transformer_path,
            paths.splits / "splits.pt",
        ],
        metadata={
            "fit_scope": "train_message_graph_only",
            "fitted_graph_hash": transformer.fitted_graph_hash,
            "transformer_state_hash": transformer.state_hash(),
            "go_vocabulary_size": len(transformer.go_vocabulary),
            "feature_shapes": feature_shapes,
        },
    )
    _log_stage(payload)
    return payload


def train_sample_baselines(
    paths: SamplePaths = DEFAULT_PATHS,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train and evaluate the executable baseline suite."""

    _validate_stage_manifest(paths.manifests / "split.json")
    _validate_stage_manifest(paths.manifests / "features.json")
    _validate_current_run(paths, require_checkpoint=False)
    nodes, reference_edges = _read_graph_tables(paths.graph)
    _, train_edges = _read_graph_tables(paths.train_graph)
    split = load_target_split(paths.splits)
    feature_path = _require_file(paths.features / "heterodata.pt")
    message_data = cast(HeteroData, torch.load(feature_path, weights_only=False))
    transformer = TrainGraphFeatureTransformer.from_dict(
        _read_json(_require_file(paths.features / "feature_transformer.json"))
    )
    baseline_dir = paths.models / "baselines"
    baseline_metrics_dir = paths.metrics / "baselines"
    sampling_dir = paths.splits / "negative_sampling"
    for output_dir in (baseline_dir, baseline_metrics_dir, sampling_dir):
        if output_dir.exists():
            shutil.rmtree(output_dir)
    results = run_baseline_suite(
        nodes=nodes,
        reference_edges=reference_edges,
        train_edges=train_edges,
        split=split,
        message_data=message_data,
        transformer=transformer,
        models_dir=baseline_dir,
        metrics_dir=baseline_metrics_dir,
        sampling_dir=sampling_dir,
        seed=seed,
    )
    outputs = [
        *sorted(baseline_dir.glob("*")),
        *sorted(baseline_metrics_dir.glob("*")),
        *sorted(sampling_dir.glob("*")),
    ]
    payload = _write_stage_manifest(
        paths,
        "train-baselines",
        inputs=[
            paths.splits / "splits.pt",
            paths.splits / "full_known_positives.parquet",
            paths.features / "feature_transformer.json",
            feature_path,
            *sorted((paths.train_graph / "edges").glob("*.parquet")),
        ],
        outputs=outputs,
        metadata={
            "models": sorted(results["models"]),
            "optional_models": results["optional_models"],
            "sampling_strategies": sorted(results["training_unlabeled_strategy"]),
            "seed": seed,
        },
    )
    _log_stage(payload)
    return payload


def train_sample_gnn(
    paths: SamplePaths = DEFAULT_PATHS,
    *,
    seed: int = SEED,
    max_epochs: int = 5,
) -> dict[str, Any]:
    """Train all configured GNN families across five fixed seeds."""

    _validate_stage_manifest(paths.manifests / "features.json")
    _validate_stage_manifest(paths.manifests / "train-baselines.json")
    _validate_current_run(paths, require_checkpoint=False)
    feature_path = _require_file(paths.features / "heterodata.pt")
    bundle = load_splits(paths.splits)
    message_data = cast(HeteroData, torch.load(feature_path, weights_only=False))
    reference = _evaluation_reference(paths, message_data)
    config_path = _sample_experiment_config_path()
    experiment = load_experiment_config(
        config_path,
        max_epochs_override=max_epochs,
    )
    if seed != experiment.seeds[0]:
        msg = (
            f"Pipeline seed {seed} must match the first configured experiment seed "
            f"{experiment.seeds[0]}"
        )
        raise ValueError(msg)
    artifact_metadata = _gnn_artifact_metadata(paths)
    baseline_results = _read_json(
        _require_file(paths.metrics / "baselines" / "results.json")
    )
    popularity = float(
        cast(dict[str, Any], baseline_results["models"])["degree_popularity"]["AUPRC"]
    )
    baseline_metrics = cast(dict[str, dict[str, Any]], baseline_results["models"])
    for output_dir in (
        paths.models / "gnn",
        paths.metrics / "gnn",
        paths.metrics / "comparisons",
    ):
        if output_dir.exists():
            shutil.rmtree(output_dir)
    comparison = run_gnn_experiment(
        bundle,
        reference,
        experiment=experiment,
        artifact_metadata=artifact_metadata,
        sampled_unlabeled=load_sampled_unlabeled(paths.splits / "negative_sampling"),
        baseline_metrics=baseline_metrics,
        popularity_auprc=popularity,
        models_dir=paths.models / "gnn",
        metrics_dir=paths.metrics / "gnn",
        comparisons_dir=paths.metrics / "comparisons",
        report_path=paths.report / "benchmark.md",
    )
    outputs = [
        *sorted((paths.models / "gnn").rglob("*.*")),
        *sorted((paths.metrics / "gnn").rglob("*.json")),
        *sorted((paths.metrics / "comparisons").glob("*.json")),
        paths.report / "benchmark.md",
    ]
    for output in outputs:
        _require_file(output)
    payload = _write_stage_manifest(
        paths,
        "train-gnn",
        inputs=[
            paths.splits / "splits.pt",
            feature_path,
            paths.splits / "full_known_positives.parquet",
            paths.metrics / "baselines" / "results.json",
            config_path,
        ],
        outputs=outputs,
        metadata={
            "models": list(experiment.models),
            "seeds": list(experiment.seeds),
            "max_epochs": max_epochs,
            "split_hash": artifact_metadata["split_hash"],
            "comparison_models": sorted(comparison["models"]),
        },
    )
    _log_stage(payload)
    return payload


def evaluate_sample(
    paths: SamplePaths = DEFAULT_PATHS,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Reload every GNN checkpoint and verify persisted held-out metrics."""

    _validate_stage_manifest(paths.manifests / "train-baselines.json")
    _validate_stage_manifest(paths.manifests / "train-gnn.json")
    _validate_current_run(paths, require_checkpoint=True)
    feature_path = _require_file(paths.features / "heterodata.pt")
    message_data = cast(HeteroData, torch.load(feature_path, weights_only=False))
    reference = _evaluation_reference(paths, message_data)
    bundle = load_splits(paths.splits)
    positives, _ = _label_triples(bundle.test_data, DISEASE_GENE_EDGE)
    if not positives:
        msg = "Evaluation requires positive held-out disease-gene links"
        raise ValueError(msg)
    baseline_results = _read_json(
        _require_file(paths.metrics / "baselines" / "results.json")
    )
    baseline_metrics = cast(dict[str, Any], baseline_results["models"])
    train_manifest = _read_json(_require_file(paths.manifests / "train-gnn.json"))
    max_epochs = int(train_manifest["metadata"]["max_epochs"])
    experiment = load_experiment_config(
        _sample_experiment_config_path(),
        max_epochs_override=max_epochs,
    )
    artifact_metadata = _gnn_artifact_metadata(paths)
    sampled = load_sampled_unlabeled(paths.splits / "negative_sampling")
    verified_checkpoints = 0
    for model_name in experiment.models:
        config = dataclasses.replace(experiment.training, model_name=model_name)
        for model_seed in experiment.seeds:
            metric_path = _require_file(
                paths.metrics / "gnn" / model_name / f"seed_{model_seed}.json"
            )
            saved = _read_json(metric_path)
            model = load_training_checkpoint(
                _require_file(
                    paths.models
                    / "gnn"
                    / model_name
                    / f"seed_{model_seed}"
                    / "best_checkpoint.pt"
                ),
                bundle.train_data.metadata(),
                config,
                expected_artifact_metadata=artifact_metadata,
            )
            reproduced = flatten_numeric_metrics(
                evaluate_split_detailed(
                    model,
                    bundle.test_data,
                    reference,
                    edge_types=config.edge_types,
                    sampled_unlabeled=sampled,
                )
            )
            expected = cast(dict[str, float], saved["flat_metrics"])
            expected_reproduced = {
                key: value
                for key, value in expected.items()
                if not key.startswith("primary.lift_over_popularity_")
            }
            if reproduced.keys() != expected_reproduced.keys() or any(
                not math.isclose(
                    reproduced[key],
                    expected_reproduced[key],
                    abs_tol=1e-7,
                )
                for key in reproduced
            ):
                msg = f"Reloaded checkpoint metrics differ for {model_name} seed {model_seed}"
                raise ValueError(msg)
            verified_checkpoints += 1

    outputs: list[Path] = []
    comparison = _read_json(
        _require_file(paths.metrics / "comparisons" / "gnn_comparison.json")
    )
    gnn_models = cast(dict[str, Any], comparison["models"])
    primary_auprc = str(comparison["primary_metric"])
    primary_auroc = primary_auprc.replace("AUPRC", "AUROC")
    primary_mrr = primary_auprc.replace("AUPRC", "filtered.MRR")
    gnn_metrics = {
        name: {
            "AUPRC": model["summary"][primary_auprc]["mean"],
            "AUROC": model["summary"][primary_auroc]["mean"],
            "filtered_MRR": model["summary"][primary_mrr]["mean"],
            "summary_scope": "five-seed full-candidate mean",
        }
        for name, model in gnn_models.items()
    }
    all_metrics: dict[str, Any] = {**baseline_metrics, **gnn_metrics}
    for name, metrics in all_metrics.items():
        output = paths.metrics / f"{name}.json"
        _write_json(output, cast(dict[str, Any], metrics))
        outputs.append(output)
    results_path = paths.metrics / "results.json"
    _write_json(
        results_path,
        {
            "dataset": "kgtp-small-oa-sample",
            "scope": "pipeline validation only; not a scientific benchmark",
            "seed": seed,
            "gnn_seeds": list(experiment.seeds),
            "test_positive_links": len(positives),
            "models": all_metrics,
            "gnn_comparison": comparison,
            "verified_checkpoints": verified_checkpoints,
        },
    )
    outputs.append(results_path)
    payload = _write_stage_manifest(
        paths,
        "evaluate",
        inputs=[
            paths.splits / "splits.pt",
            paths.metrics / "baselines" / "results.json",
            paths.metrics / "comparisons" / "gnn_comparison.json",
        ],
        outputs=outputs,
        metadata={
            "models": sorted(all_metrics),
            "test_positive_links": len(positives),
            "verified_checkpoints": verified_checkpoints,
        },
    )
    _log_stage(payload)
    return payload


def report_sample(paths: SamplePaths = DEFAULT_PATHS) -> dict[str, Any]:
    """Write a compact machine-readable and Markdown sample-run report."""

    _require_file(paths.manifests / "evaluate.json")
    source = _read_json(_require_file(paths.manifests / "source_manifest.json"))
    statistics = _read_json(_require_file(paths.graph / "statistics.json"))
    results = _read_json(_require_file(paths.metrics / "results.json"))
    report_json = paths.report / "report.json"
    report_md = paths.report / "report.md"
    report_payload = {
        "dataset": source.get("dataset"),
        "retrieval_date": source.get("retrieval_date"),
        "graph": statistics,
        "evaluation": results,
        "limitations": [
            "The checked-in sample is intentionally small and incomplete.",
            "Metrics validate execution and must not be interpreted as biomedical evidence.",
            "The protocol is transductive for non-target relations and node identities.",
            "Unlabeled pairs may contain unknown biological positives absent from the registry.",
        ],
    }
    _write_json(report_json, report_payload)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        _render_report(source, statistics, results),
        encoding="utf-8",
    )
    payload = _write_stage_manifest(
        paths,
        "report",
        inputs=[
            paths.manifests / "source_manifest.json",
            paths.graph / "statistics.json",
            paths.metrics / "results.json",
        ],
        outputs=[report_json, report_md],
        metadata={"format": ["json", "markdown"]},
    )
    _log_stage(payload)
    return payload


def reproduce_small(
    paths: SamplePaths = DEFAULT_PATHS,
    *,
    seed: int = SEED,
    max_epochs: int = 5,
) -> None:
    """Execute every required sample stage, failing on the first invalid stage."""
    run_manifest = paths.manifests / "run.json"
    config_path = _sample_experiment_config_path()
    start_run_manifest(
        run_manifest,
        sample_manifest_path=paths.sample_dir / "manifest.json",
        config_path=config_path,
        seed=seed,
        max_epochs=max_epochs,
        command=default_command(seed, max_epochs),
    )
    try:
        prepare_sample(paths)
        normalize_sample(paths)
        update_run_manifest(
            run_manifest,
            normalized_file_hashes=hashes_for_files(
                list(paths.normalized.glob("*.parquet"))
            ),
        )
        assemble_sample(paths)
        split_sample(paths, seed=seed)
        split_metadata = _read_json(paths.splits / "split_metadata.json")
        update_run_manifest(
            run_manifest,
            graph_hash=str(split_metadata["full_reference_graph_hash"]),
            split_hash=sha256_file(paths.splits / "split_metadata.json"),
        )
        build_sample_features(paths)
        artifact_metadata = _gnn_artifact_metadata(paths)
        update_run_manifest(
            run_manifest,
            feature_hash=artifact_metadata["feature_transformer_hash"],
        )
        train_sample_baselines(paths, seed=seed)
        train_sample_gnn(paths, seed=seed, max_epochs=max_epochs)
        experiment = load_experiment_config(
            config_path,
            max_epochs_override=max_epochs,
        )
        checkpoint_paths = sorted(
            (paths.models / "gnn").glob("*/seed_*/best_checkpoint.pt")
        )
        primary_checkpoint = (
            paths.models / "gnn" / "hgt" / f"seed_{seed}" / "best_checkpoint.pt"
        )
        update_run_manifest(
            run_manifest,
            checkpoint_hash=sha256_file(primary_checkpoint),
            checkpoint_hashes=hashes_for_files(checkpoint_paths),
            seeds=list(experiment.seeds),
        )
        evaluate_sample(paths, seed=seed)
        report_sample(paths)
        complete_run_manifest(run_manifest)
    except Exception as exc:
        fail_run_manifest(run_manifest, exc)
        raise


def clean_sample(paths: SamplePaths = DEFAULT_PATHS) -> None:
    """Remove only the configured sample artifact root."""

    target = paths.root.resolve()
    if target.name != "sample" or target == target.parent:
        msg = f"Refusing to remove unsafe sample artifact path: {target}"
        raise ValueError(msg)
    if target.exists():
        shutil.rmtree(target)
    print(json.dumps({"stage": "clean-sample", "removed": str(target)}, sort_keys=True))


def _normalize_tables(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    nodes = tables["nodes.parquet"].copy()
    ensure_columns(nodes, ("node_id", "node_type", "label"))
    gene_mask = nodes["node_type"].astype(str) == "Gene"
    nodes.loc[gene_mask, "node_id"] = nodes.loc[gene_mask, "node_id"].map(
        normalize_ensembl_gene_id
    )
    nodes = nodes.dropna(subset=["node_id", "node_type"])
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["node_type"] = nodes["node_type"].astype(str)
    nodes = stable_drop_duplicates(nodes, ["node_type", "node_id"])

    disease_gene = _normalize_gene_columns(
        tables["disease_gene.parquet"],
        required=("disease_id", "gene_id", "score"),
        gene_columns=("gene_id",),
        keys=("disease_id", "gene_id"),
    )
    gene_gene = _normalize_gene_columns(
        tables["gene_gene.parquet"],
        required=("gene_a", "gene_b", "score"),
        gene_columns=("gene_a", "gene_b"),
        keys=("gene_a", "gene_b"),
    )
    canonical_pairs = [
        tuple(sorted((str(first), str(second))))
        for first, second in zip(gene_gene["gene_a"], gene_gene["gene_b"], strict=True)
    ]
    gene_gene["gene_a"] = [pair[0] for pair in canonical_pairs]
    gene_gene["gene_b"] = [pair[1] for pair in canonical_pairs]
    gene_gene = gene_gene[gene_gene["gene_a"] != gene_gene["gene_b"]]
    gene_gene = stable_drop_duplicates(gene_gene, ["gene_a", "gene_b"])

    gene_pathway = _normalize_gene_columns(
        tables["gene_pathway.parquet"],
        required=("gene_id", "pathway_id", "pathway_name"),
        gene_columns=("gene_id",),
        keys=("gene_id", "pathway_id"),
    )
    drug_gene = _normalize_gene_columns(
        tables["drug_gene.parquet"],
        required=("drug_id", "gene_id", "target_chembl_id"),
        gene_columns=("gene_id",),
        keys=("drug_id", "gene_id", "action_type"),
    )
    gene_go = _normalize_gene_columns(
        tables["gene_go.parquet"],
        required=("gene_id", "go_id", "evidence_code"),
        gene_columns=("gene_id",),
        keys=("gene_id", "go_id", "evidence_code"),
    )
    pathway_pathway = tables["pathway_pathway.parquet"].copy()
    ensure_columns(pathway_pathway, ("parent_pathway_id", "child_pathway_id"))
    pathway_pathway = stable_drop_duplicates(
        pathway_pathway, ["parent_pathway_id", "child_pathway_id"]
    )
    normalized = {
        "nodes.parquet": nodes,
        "disease_gene.parquet": disease_gene,
        "gene_gene.parquet": gene_gene,
        "gene_pathway.parquet": gene_pathway,
        "drug_gene.parquet": drug_gene,
        "gene_go.parquet": gene_go,
        "pathway_pathway.parquet": pathway_pathway,
    }
    for filename, frame in normalized.items():
        if frame.empty:
            msg = f"Normalization produced an empty required table: {filename}"
            raise ValueError(msg)
    return normalized


def _normalize_gene_columns(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    gene_columns: Sequence[str],
    keys: Sequence[str],
) -> pd.DataFrame:
    output = frame.copy()
    ensure_columns(output, required)
    for column in gene_columns:
        output[column] = output[column].map(normalize_ensembl_gene_id)
    output = output.dropna(subset=list(gene_columns))
    for column in gene_columns:
        output[column] = output[column].astype(str)
    return stable_drop_duplicates(output, keys)


def _validate_graph_endpoints(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
) -> None:
    known = {
        (str(node_type), str(node_id))
        for node_type, node_id in zip(nodes["node_type"], nodes["node_id"], strict=True)
    }
    missing: list[str] = []
    for name, edges in edge_tables.items():
        for row in edges.itertuples(index=False):
            source = (str(row.source_type), str(row.source_id))
            target = (str(row.target_type), str(row.target_id))
            if source not in known:
                missing.append(f"{name}: source {source}")
            if target not in known:
                missing.append(f"{name}: target {target}")
    if missing:
        msg = "Graph endpoints missing from nodes table: " + "; ".join(missing[:10])
        raise ValueError(msg)


def _read_graph_tables(
    graph_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    nodes = read_table(_require_file(graph_dir / "nodes.parquet"))
    edge_paths = sorted((graph_dir / "edges").glob("*.parquet"))
    if not edge_paths:
        msg = f"No canonical edge tables found under {graph_dir / 'edges'}"
        raise FileNotFoundError(msg)
    edge_tables = {path.stem: read_table(path) for path in edge_paths}
    assert_canonical_edge_types(edge_tables)
    _validate_graph_endpoints(nodes, edge_tables)
    return nodes, edge_tables


def _evaluation_reference(
    paths: SamplePaths,
    message_data: HeteroData,
) -> EvaluationReference:
    _, edge_tables = _read_graph_tables(paths.graph)
    task_tables = {
        DISEASE_GENE_EDGE: "disease_gene",
        ("drug", "targets", "gene"): "drug_gene",
        ("gene", "participates_in", "pathway"): "gene_pathway",
    }
    return EvaluationReference(
        known_triples={
            edge_type: {
                (str(row.source_id), edge_type[1], str(row.target_id))
                for row in edge_tables[table].itertuples(index=False)
            }
            for edge_type, table in task_tables.items()
        },
        node_ids={
            node_type: _node_ids(message_data, node_type)
            for node_type in message_data.node_types
        },
    )


def _label_triples(
    data: HeteroData,
    edge_type: tuple[str, str, str],
) -> tuple[list[Triple], list[Triple]]:
    source_type, relation, target_type = edge_type
    source_ids = _node_ids(data, source_type)
    target_ids = _node_ids(data, target_type)
    labels = data[edge_type].edge_label.tolist()
    pairs = data[edge_type].edge_label_index.t().tolist()
    positives: list[Triple] = []
    negatives: list[Triple] = []
    for (source, target), label in zip(pairs, labels, strict=True):
        triple = (
            source_ids[int(source)],
            relation,
            target_ids[int(target)],
        )
        (positives if float(label) == 1.0 else negatives).append(triple)
    return positives, negatives


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    return [str(value) for value in data[node_type].node_id]


def _x_dict(data: HeteroData) -> dict[str, torch.Tensor]:
    return {node_type: data[node_type].x for node_type in data.node_types}


def _edge_index_dict(
    data: HeteroData,
) -> dict[tuple[str, str, str], torch.Tensor]:
    return {
        edge_type: data[edge_type].edge_index
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }


def _gnn_artifact_metadata(paths: SamplePaths) -> dict[str, str]:
    split_metadata = _read_json(_require_file(paths.splits / "split_metadata.json"))
    feature_metadata = _read_json(
        _require_file(paths.features / "feature_transformer.json")
    )
    return {
        "dataset_hash": str(split_metadata["full_reference_graph_hash"]),
        "split_hash": _sha256(paths.splits / "split_metadata.json"),
        "train_message_graph_hash": str(split_metadata["train_message_graph_hash"]),
        "node_index_map_hash": str(split_metadata["node_index_map_hash"]),
        "feature_transformer_hash": hashlib.sha256(
            json.dumps(feature_metadata, sort_keys=True).encode()
        ).hexdigest(),
    }


def _sample_experiment_config_path() -> Path:
    path = Path("configs/sample-gnn.yaml").resolve()
    if not path.is_file():
        msg = (
            "Missing configs/sample-gnn.yaml; run the sample pipeline from the "
            "repository root"
        )
        raise FileNotFoundError(msg)
    return path


def _render_report(
    source: Mapping[str, Any],
    statistics: Mapping[str, Any],
    results: Mapping[str, Any],
) -> str:
    models = cast(dict[str, dict[str, Any]], results["models"])
    lines = [
        "# Small OA sample pipeline report",
        "",
        "This report validates executable repository plumbing. It is not a "
        "scientific benchmark and must not be used as biomedical evidence.",
        "",
        f"- Dataset: `{source['dataset']}`",
        f"- Snapshot retrieval date: `{source['retrieval_date']}`",
        f"- Nodes by type: `{json.dumps(statistics['node_counts'], sort_keys=True)}`",
        f"- Disease-gene positives: `{statistics['positive_disease_gene_links']}`",
        f"- Test positive links: `{results['test_positive_links']}`",
        "",
        "## Execution metrics",
        "",
        "| Model | AUPRC | AUROC | Filtered MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(models):
        metrics = models[name]
        filtered = cast(dict[str, Any], metrics.get("filtered", {}))
        mrr = metrics.get("filtered_MRR", filtered.get("MRR"))
        lines.append(
            f"| {name} | {_format_metric(metrics.get('AUPRC'))} | "
            f"{_format_metric(metrics.get('AUROC'))} | {_format_metric(mrr)} |"
        )
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- The sample is intentionally small and incomplete.",
            "- Metrics demonstrate execution only; they are not efficacy claims.",
            "- Target edges are split before topology-derived feature fitting.",
            "- Non-target relations and the node universe are shared transductively.",
            "- Unknown positives can still be sampled as unlabeled when absent "
            "from the known-positive registry.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(cast(float, value)):.4f}"


def _write_stage_manifest(
    paths: SamplePaths,
    stage: str,
    *,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    for path in outputs:
        _require_file(path)
    payload = {
        "schema_version": 1,
        "stage": stage,
        "status": "completed",
        "inputs": [_file_record(path) for path in inputs],
        "outputs": [_file_record(path) for path in outputs],
        "metadata": _json_ready(metadata),
    }
    _write_json(paths.manifests / f"{stage}.json", payload)
    return payload


def _validate_stage_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(_require_file(path))
    if payload.get("status") != "completed":
        msg = f"Stage manifest is not completed: {path}"
        raise ValueError(msg)
    for section in ("inputs", "outputs"):
        records = payload.get(section)
        if not isinstance(records, list) or not records:
            msg = f"Stage manifest has no {section}: {path}"
            raise ValueError(msg)
        for record in records:
            if not isinstance(record, dict):
                msg = f"Invalid {section} record in {path}"
                raise ValueError(msg)
            artifact = Path(str(record.get("path", "")))
            if not artifact.is_absolute():
                artifact = Path.cwd() / artifact
            if not artifact.is_file():
                msg = f"Manifest artifact is missing: {artifact}"
                raise FileNotFoundError(msg)
            if _sha256(artifact) != record.get("sha256"):
                msg = f"Manifest artifact hash mismatch: {artifact}"
                raise ValueError(msg)
    return payload


def _validate_current_run(paths: SamplePaths, *, require_checkpoint: bool) -> None:
    run_path = paths.manifests / "run.json"
    if not run_path.is_file():
        return
    metadata = _gnn_artifact_metadata(paths)
    checkpoint_hash = None
    if require_checkpoint:
        checkpoint_hash = sha256_file(
            paths.models / "gnn" / "hgt" / "seed_13" / "best_checkpoint.pt"
        )
    validate_run_compatibility(
        run_path,
        graph_hash=metadata["dataset_hash"],
        split_hash=metadata["split_hash"],
        feature_hash=metadata["feature_transformer_hash"],
        checkpoint_hash=checkpoint_hash,
    )


def _file_record(path: Path) -> dict[str, Any]:
    checked = _require_file(path)
    return {
        "path": checked.as_posix(),
        "bytes": checked.stat().st_size,
        "sha256": _sha256(checked),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        msg = f"Required file does not exist: {path}"
        raise FileNotFoundError(msg)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}"
        raise TypeError(msg)
    return cast(dict[str, Any], payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _log_stage(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "stage": payload["stage"],
                "status": payload["status"],
                "outputs": len(cast(list[Any], payload["outputs"])),
            },
            sort_keys=True,
        )
    )
