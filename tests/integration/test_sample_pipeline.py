"""End-to-end coverage for the checked-in sample pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from kgtp.artifacts import ArtifactPaths, validate_and_load_artifacts
from kgtp.explain.runner import run_validated_explanations
from kgtp.pipeline.sample import SamplePaths, reproduce_small


def test_sample_pipeline_writes_complete_artifact_tree(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
    paths = SamplePaths(
        sample_dir=repository_root / "data" / "sample",
        root=tmp_path / "sample",
    )

    reproduce_small(paths, seed=13, max_epochs=2)

    expected = (
        paths.manifests / "prepare-sample.json",
        paths.manifests / "normalize.json",
        paths.manifests / "assemble.json",
        paths.manifests / "dataset.json",
        paths.manifests / "features.json",
        paths.manifests / "split.json",
        paths.manifests / "train-baselines.json",
        paths.manifests / "train-gnn.json",
        paths.manifests / "evaluate.json",
        paths.manifests / "report.json",
        paths.manifests / "run.json",
        paths.normalized / "nodes.parquet",
        paths.graph / "nodes.parquet",
        paths.graph / "edges" / "disease_gene.parquet",
        paths.train_graph / "nodes.parquet",
        paths.train_graph / "edges" / "disease_gene.parquet",
        paths.splits / "full_known_positives.parquet",
        paths.splits / "split_assignments.parquet",
        paths.splits / "leakage_validation.json",
        paths.splits / "supervision" / "train.parquet",
        paths.splits / "supervision" / "validation.parquet",
        paths.splits / "supervision" / "test.parquet",
        paths.features / "heterodata.pt",
        paths.features / "feature_transformer.json",
        paths.splits / "splits.pt",
        paths.models / "baselines" / "adjacency_svd.json",
        paths.models / "baselines" / "node2vec.json",
        paths.models / "baselines" / "distmult.pt",
        paths.models / "baselines" / "complex.pt",
        paths.models / "baselines" / "availability.json",
        paths.models / "baselines" / "matrix_factorization.npz",
        paths.metrics / "baselines" / "results.json",
        paths.splits / "negative_sampling" / "random.parquet",
        paths.splits / "negative_sampling" / "degree_matched.parquet",
        paths.splits / "negative_sampling" / "hard.parquet",
        paths.models / "gnn" / "hgt" / "seed_13" / "best_checkpoint.pt",
        paths.models / "gnn" / "hgt" / "seed_13" / "config.json",
        paths.models / "gnn" / "graphsage" / "seed_17" / "best_checkpoint.pt",
        paths.models
        / "gnn"
        / "graphsage_homogeneous"
        / "seed_19"
        / "best_checkpoint.pt",
        paths.models / "gnn" / "rgcn" / "seed_23" / "best_checkpoint.pt",
        paths.metrics / "gnn" / "hgt" / "summary.json",
        paths.metrics / "gnn" / "graphsage" / "summary.json",
        paths.metrics / "gnn" / "graphsage_homogeneous" / "summary.json",
        paths.metrics / "gnn" / "rgcn" / "summary.json",
        paths.metrics / "comparisons" / "gnn_comparison.json",
        paths.metrics / "results.json",
        paths.report / "benchmark.md",
        paths.report / "report.md",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)

    results = json.loads((paths.metrics / "results.json").read_text(encoding="utf-8"))
    assert results["scope"] == "pipeline validation only; not a scientific benchmark"
    assert results["test_positive_links"] > 0
    assert results["verified_checkpoints"] == 20
    assert {
        "hgt",
        "matrix_factorization",
        "adjacency_svd",
        "node2vec",
        "distmult",
        "complex",
        "hash_text",
        "feature_mlp",
        "graphsage",
        "graphsage_homogeneous",
        "rgcn",
    }.issubset(results["models"])
    run_manifest = json.loads(
        (paths.manifests / "run.json").read_text(encoding="utf-8")
    )
    archived_run = paths.manifests / "runs" / f"{run_manifest['run_id']}.json"
    assert archived_run.is_file()
    assert json.loads(archived_run.read_text(encoding="utf-8")) == run_manifest
    assert run_manifest["status"] == "completed"
    assert isinstance(run_manifest["dirty_worktree"], bool)
    assert run_manifest["git_commit"]
    assert run_manifest["dependency_lock_sha256"]
    assert run_manifest["config_sha256"]
    assert run_manifest["source_versions"]
    assert run_manifest["source_licenses"]
    assert len(run_manifest["raw_file_hashes"]) == 7
    assert len(run_manifest["normalized_file_hashes"]) == 7
    assert run_manifest["graph_hash"]
    assert run_manifest["split_hash"]
    assert run_manifest["feature_hash"]
    assert run_manifest["checkpoint_hash"]
    assert run_manifest["seeds"] == [13, 17, 19, 23, 29]

    artifacts = validate_and_load_artifacts(
        ArtifactPaths(
            checkpoint=paths.models / "gnn" / "hgt" / "seed_13" / "best_checkpoint.pt",
            model_config=paths.models / "gnn" / "hgt" / "seed_13" / "config.json",
            graph=paths.features / "heterodata.pt",
            dataset_manifest=paths.manifests / "dataset.json",
            graph_manifest=paths.manifests / "assemble.json",
            feature_manifest=paths.manifests / "features.json",
            node_index_map=paths.features / "node_index_maps.json",
            split_metadata=paths.splits / "split_metadata.json",
            validation_metrics=paths.models
            / "gnn"
            / "hgt"
            / "seed_13"
            / "metrics.json",
            run_manifest=paths.manifests / "run.json",
        ),
        root=repository_root,
    )
    explanation_result = run_validated_explanations(
        artifacts,
        paths.report / "explanations",
        integration_steps=1,
    )
    assert explanation_result["explanation_count"] == 5
    assert (paths.report / "explanations" / "ranking.json").is_file()
    assert (paths.report / "explanations" / "evidence_cards.json").is_file()
