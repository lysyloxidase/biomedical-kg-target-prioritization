from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kgtp.artifacts import ArtifactPaths  # noqa: E402
from kgtp.explain.explainer import DISEASE_GENE_EDGE  # noqa: E402
from kgtp.hetero.splits import disjoint_random_link_split  # noqa: E402
from kgtp.models.train import TrainingConfig, train_one_seed  # noqa: E402
from kgtp.run_manifest import write_run_manifest  # noqa: E402
from kgtp.smoke import tiny_heterodata  # noqa: E402


@pytest.fixture(scope="session")
def trained_artifact_paths(tmp_path_factory: pytest.TempPathFactory) -> ArtifactPaths:
    """Build a compact, internally compatible trained-artifact bundle."""
    root = tmp_path_factory.mktemp("trained-artifacts")
    model_dir = root / "models" / "hgt" / "seed_13"
    features = root / "features"
    manifests = root / "manifests"
    splits = root / "splits"
    supervision = splits / "supervision"
    for directory in (model_dir, features, manifests, supervision):
        directory.mkdir(parents=True, exist_ok=True)

    reference = tiny_heterodata()
    reference.is_smoke_test_graph = False
    bundle = disjoint_random_link_split(
        reference,
        seed=13,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.5,
        train_neg_sampling_ratio=1.0,
        eval_neg_sampling_ratio=1.0,
    )
    graph = bundle.train_data
    graph.is_smoke_test_graph = False
    graph_path = features / "heterodata.pt"
    torch.save(graph, graph_path)

    node_map = {
        node_type: {
            str(node_id): index
            for index, node_id in enumerate(graph[node_type].node_id)
        }
        for node_type in graph.node_types
    }
    node_map_path = features / "node_index_maps.json"
    _write_json(node_map_path, node_map)
    node_map_hash = _canonical_hash(node_map, compact=True)

    dataset_hash = "test-full-reference-graph"
    train_hash = "test-train-message-graph"
    split_metadata = {
        "schema_version": 1,
        "seed": 13,
        "full_reference_graph_hash": dataset_hash,
        "train_message_graph_hash": train_hash,
        "node_index_map_hash": node_map_hash,
    }
    split_path = splits / "split_metadata.json"
    _write_json(split_path, split_metadata)
    _write_supervision(bundle.train_data, supervision / "train.parquet", graph)
    _write_supervision(
        bundle.val_data,
        supervision / "validation.parquet",
        graph,
    )
    _write_supervision(bundle.test_data, supervision / "test.parquet", graph)

    transformer = {
        "schema_version": 1,
        "transformer": "TestTrainGraphFeatureTransformer",
        "fit_scope": "train_message_graph_only",
        "fitted_graph_hash": train_hash,
    }
    transformer_path = features / "feature_transformer.json"
    _write_json(transformer_path, transformer)
    transformer_hash = _canonical_hash(transformer, compact=False)

    dummy_input = root / "source.txt"
    dummy_input.write_text("test source\n", encoding="utf-8")
    graph_manifest_path = manifests / "assemble.json"
    _write_json(
        graph_manifest_path,
        {
            "schema_version": 1,
            "inputs": [_record(dummy_input)],
            "outputs": [_record(graph_path)],
        },
    )
    feature_manifest_path = manifests / "features.json"
    _write_json(
        feature_manifest_path,
        {
            "schema_version": 1,
            "inputs": [_record(dummy_input)],
            "outputs": [
                _record(graph_path),
                _record(node_map_path),
                _record(transformer_path),
            ],
            "metadata": {
                "fit_scope": "train_message_graph_only",
                "fitted_graph_hash": train_hash,
            },
        },
    )
    dataset_manifest_path = manifests / "dataset.json"
    _write_json(
        dataset_manifest_path,
        {
            "schema_version": 1,
            "dataset_id": "test-trained-artifacts",
            "dataset_version": "1",
            "full_reference_graph_hash": dataset_hash,
            "train_message_graph_hash": train_hash,
            "node_index_map_hash": node_map_hash,
        },
    )

    artifact_metadata = {
        "dataset_hash": dataset_hash,
        "split_hash": _sha256(split_path),
        "train_message_graph_hash": train_hash,
        "node_index_map_hash": node_map_hash,
        "feature_transformer_hash": transformer_hash,
    }
    config = TrainingConfig(
        model_name="hgt",
        hidden_channels=8,
        num_heads=2,
        num_layers=1,
        max_epochs=1,
        patience=1,
        edge_types=(DISEASE_GENE_EDGE,),
    )
    train_one_seed(
        graph,
        bundle.val_data,
        bundle.test_data,
        reference,
        seed=13,
        config=config,
        output_dir=model_dir,
        artifact_metadata=artifact_metadata,
    )
    run_manifest_path = manifests / "run.json"
    write_run_manifest(
        run_manifest_path,
        {
            "schema_version": 1,
            "run_id": "test-run-13",
            "git_commit": "test",
            "dirty_worktree": False,
            "created_at": "2026-06-12T00:00:00+00:00",
            "python_version": "test",
            "platform": "test",
            "dependency_lock_sha256": "test",
            "config_sha256": "test",
            "source_versions": {},
            "source_licenses": {},
            "raw_file_hashes": {},
            "normalized_file_hashes": {"test": "test"},
            "graph_hash": dataset_hash,
            "split_hash": artifact_metadata["split_hash"],
            "feature_hash": transformer_hash,
            "checkpoint_hash": _sha256(model_dir / "best_checkpoint.pt"),
            "checkpoint_hashes": {},
            "seeds": [13],
            "command": "pytest fixture",
            "status": "completed",
        },
    )
    return ArtifactPaths(
        checkpoint=model_dir / "best_checkpoint.pt",
        model_config=model_dir / "config.json",
        graph=graph_path,
        dataset_manifest=dataset_manifest_path,
        graph_manifest=graph_manifest_path,
        feature_manifest=feature_manifest_path,
        node_index_map=node_map_path,
        split_metadata=split_path,
        validation_metrics=model_dir / "metrics.json",
        run_manifest=run_manifest_path,
    )


def _write_supervision(data: Any, path: Path, graph: Any) -> None:
    store = data[DISEASE_GENE_EDGE]
    disease_ids = [str(value) for value in graph["disease"].node_id]
    gene_ids = [str(value) for value in graph["gene"].node_id]
    rows = []
    for position, label in enumerate(store.edge_label.tolist()):
        source = int(store.edge_label_index[0, position])
        target = int(store.edge_label_index[1, position])
        rows.append(
            {
                "source_id": disease_ids[source],
                "target_id": gene_ids[target],
                "label": int(label),
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any, *, compact: bool) -> str:
    separators = (",", ":") if compact else None
    encoded = json.dumps(payload, sort_keys=True, separators=separators).encode()
    return hashlib.sha256(encoded).hexdigest()
