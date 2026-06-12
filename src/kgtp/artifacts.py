"""Validation and loading of trained runtime artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from kgtp.models.train import TrainingConfig, load_training_checkpoint
from kgtp.run_manifest import (
    RunManifestError,
    load_run_manifest,
    validate_run_compatibility,
)


class ArtifactValidationError(RuntimeError):
    """Raised when runtime artifacts are missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class ArtifactPaths:
    checkpoint: Path
    model_config: Path
    graph: Path
    dataset_manifest: Path
    graph_manifest: Path
    feature_manifest: Path
    node_index_map: Path
    split_metadata: Path
    validation_metrics: Path
    run_manifest: Path


@dataclass
class ValidatedArtifacts:
    paths: ArtifactPaths
    data: HeteroData
    model: torch.nn.Module
    training_config: TrainingConfig
    model_config: dict[str, Any]
    dataset_manifest: dict[str, Any]
    graph_manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    node_index_map: dict[str, dict[str, int]]
    split_metadata: dict[str, Any]
    validation_metrics: dict[str, Any]
    run_manifest: dict[str, Any]
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    artifact_metadata: dict[str, str]


def default_sample_artifact_paths(root: Path | None = None) -> ArtifactPaths:
    base = (root or Path.cwd()) / "artifacts" / "sample"
    model_dir = base / "models" / "gnn" / "hgt" / "seed_13"
    return ArtifactPaths(
        checkpoint=model_dir / "best_checkpoint.pt",
        model_config=model_dir / "config.json",
        graph=base / "features" / "heterodata.pt",
        dataset_manifest=base / "manifests" / "dataset.json",
        graph_manifest=base / "manifests" / "assemble.json",
        feature_manifest=base / "manifests" / "features.json",
        node_index_map=base / "features" / "node_index_maps.json",
        split_metadata=base / "splits" / "split_metadata.json",
        validation_metrics=model_dir / "metrics.json",
        run_manifest=base / "manifests" / "run.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any, *, compact: bool = True) -> str:
    separators = (",", ":") if compact else None
    encoded = json.dumps(payload, sort_keys=True, separators=separators).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_files(paths: ArtifactPaths) -> None:
    for label, path in vars(paths).items():
        if not path.is_file():
            raise ArtifactValidationError(
                f"Required {label.replace('_', ' ')} is missing: {path}"
            )


def _resolve_record_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _verify_manifest_records(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    sections: tuple[str, ...],
    root: Path,
) -> None:
    for section in sections:
        records = manifest.get(section)
        if not isinstance(records, list) or not records:
            raise ArtifactValidationError(
                f"{manifest_path} has no non-empty '{section}' artifact records"
            )
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ArtifactValidationError(
                    f"Invalid artifact record in {manifest_path}"
                )
            path = _resolve_record_path(record["path"], root)
            expected_hash = record.get("sha256")
            if not path.is_file():
                raise ArtifactValidationError(
                    f"Manifest artifact is missing: {path} (declared by {manifest_path})"
                )
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ArtifactValidationError(
                    f"Manifest hash mismatch for {path} (declared by {manifest_path})"
                )


def _training_config(payload: dict[str, Any]) -> TrainingConfig:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ArtifactValidationError("Model configuration has no 'config' object")
    normalized = dict(config_payload)
    edge_types = normalized.get("edge_types")
    if isinstance(edge_types, list):
        normalized["edge_types"] = tuple(tuple(edge_type) for edge_type in edge_types)
    try:
        return TrainingConfig(**normalized)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Invalid model training configuration: {exc}"
        ) from exc


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ArtifactValidationError(
            f"Artifact compatibility check failed for {label}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_and_load_artifacts(
    paths: ArtifactPaths,
    *,
    root: Path | None = None,
) -> ValidatedArtifacts:
    """Validate hashes and compatibility before loading a trained model."""
    _require_files(paths)
    artifact_root = (root or Path.cwd()).resolve()

    model_config = _load_json(paths.model_config, "model configuration")
    dataset_manifest = _load_json(paths.dataset_manifest, "dataset manifest")
    graph_manifest = _load_json(paths.graph_manifest, "graph manifest")
    feature_manifest = _load_json(paths.feature_manifest, "feature manifest")
    node_index_map = _load_json(paths.node_index_map, "node-index map")
    split_metadata = _load_json(paths.split_metadata, "split metadata")
    validation_metrics = _load_json(paths.validation_metrics, "validation metrics")
    try:
        run_manifest = load_run_manifest(paths.run_manifest)
    except RunManifestError as exc:
        raise ArtifactValidationError(str(exc)) from exc
    training_config = _training_config(model_config)

    _verify_manifest_records(
        graph_manifest,
        paths.graph_manifest,
        sections=("inputs", "outputs"),
        root=artifact_root,
    )
    _verify_manifest_records(
        feature_manifest,
        paths.feature_manifest,
        sections=("inputs", "outputs"),
        root=artifact_root,
    )

    if model_config.get("trained") is not True:
        raise ArtifactValidationError("Model configuration is not marked as trained")
    metrics = validation_metrics.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ArtifactValidationError("Validation metadata contains no metrics")

    try:
        checkpoint = torch.load(
            paths.checkpoint, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise ArtifactValidationError(f"Checkpoint loading failed: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ArtifactValidationError("Checkpoint payload must be a mapping")

    artifact_metadata = checkpoint.get("artifact_metadata")
    if not isinstance(artifact_metadata, dict):
        raise ArtifactValidationError(
            "Checkpoint has no artifact compatibility metadata"
        )
    required_hashes = {
        "dataset_hash",
        "split_hash",
        "train_message_graph_hash",
        "node_index_map_hash",
        "feature_transformer_hash",
    }
    missing_hashes = required_hashes.difference(artifact_metadata)
    if missing_hashes:
        raise ArtifactValidationError(
            f"Checkpoint is missing compatibility hashes: {sorted(missing_hashes)}"
        )

    config_metadata = model_config.get("artifact_metadata")
    _assert_equal("model-config metadata", config_metadata, artifact_metadata)
    _assert_equal(
        "validation-metrics metadata",
        validation_metrics.get("artifact_metadata"),
        artifact_metadata,
    )
    checkpoint_config = _training_config({"config": checkpoint.get("config")})
    _assert_equal("model configuration", training_config, checkpoint_config)

    split_hash = sha256_file(paths.split_metadata)
    _assert_equal("split metadata hash", split_hash, artifact_metadata["split_hash"])
    _assert_equal(
        "dataset full-reference graph hash",
        dataset_manifest.get("full_reference_graph_hash"),
        artifact_metadata["dataset_hash"],
    )
    _assert_equal(
        "dataset train-message graph hash",
        dataset_manifest.get("train_message_graph_hash"),
        artifact_metadata["train_message_graph_hash"],
    )
    _assert_equal(
        "split full-reference graph hash",
        split_metadata.get("full_reference_graph_hash"),
        artifact_metadata["dataset_hash"],
    )
    _assert_equal(
        "split train-message graph hash",
        split_metadata.get("train_message_graph_hash"),
        artifact_metadata["train_message_graph_hash"],
    )

    node_map_hash = _canonical_hash(node_index_map)
    _assert_equal(
        "node-index map semantic hash",
        node_map_hash,
        artifact_metadata["node_index_map_hash"],
    )
    _assert_equal(
        "dataset node-index map hash",
        dataset_manifest.get("node_index_map_hash"),
        artifact_metadata["node_index_map_hash"],
    )

    feature_metadata = feature_manifest.get("metadata")
    if not isinstance(feature_metadata, dict):
        raise ArtifactValidationError("Feature manifest has no metadata object")
    _assert_equal(
        "feature fitting graph",
        feature_metadata.get("fitted_graph_hash"),
        artifact_metadata["train_message_graph_hash"],
    )
    transformer_path = (
        paths.feature_manifest.parent.parent / "features" / "feature_transformer.json"
    )
    if not transformer_path.is_file():
        raise ArtifactValidationError(
            f"Feature transformer is missing: {transformer_path}"
        )
    transformer = _load_json(transformer_path, "feature transformer")
    _assert_equal(
        "feature-transformer semantic hash",
        _canonical_hash(transformer, compact=False),
        artifact_metadata["feature_transformer_hash"],
    )
    try:
        validate_run_compatibility(
            paths.run_manifest,
            graph_hash=str(artifact_metadata["dataset_hash"]),
            split_hash=str(artifact_metadata["split_hash"]),
            feature_hash=str(artifact_metadata["feature_transformer_hash"]),
            checkpoint_hash=sha256_file(paths.checkpoint),
            allowed_statuses=("completed",),
        )
    except RunManifestError as exc:
        raise ArtifactValidationError(str(exc)) from exc
    validation_seed = validation_metrics.get("seed")
    if validation_seed not in run_manifest.get("seeds", []):
        raise ArtifactValidationError(
            f"Validation seed {validation_seed!r} is absent from the run manifest"
        )

    try:
        data = torch.load(paths.graph, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactValidationError(f"Graph loading failed: {exc}") from exc
    if not isinstance(data, HeteroData):
        raise ArtifactValidationError("Graph artifact is not a PyG HeteroData object")
    if bool(getattr(data, "is_smoke_test_graph", False)):
        raise ArtifactValidationError(
            "Smoke-test graphs cannot be used by the production API"
        )

    _attach_reference_node_metadata(data, graph_manifest, artifact_root)
    for node_type in data.node_types:
        actual_ids = list(getattr(data[node_type], "node_id", []))
        actual_map = {str(node_id): index for index, node_id in enumerate(actual_ids)}
        if node_index_map.get(node_type) != actual_map:
            raise ArtifactValidationError(
                f"Node-index map mismatch for node type '{node_type}'"
            )

    disease_map = node_index_map.get("disease", {})
    gene_map = node_index_map.get("gene", {})
    message_pairs = {
        (int(source), int(target))
        for source, target in data[("disease", "associated_with", "gene")]
        .edge_index.t()
        .tolist()
    }
    supervision_dir = paths.split_metadata.parent / "supervision"
    for partition in ("train", "validation", "test"):
        partition_path = supervision_dir / f"{partition}.parquet"
        if not partition_path.is_file():
            raise ArtifactValidationError(
                f"Required {partition} supervision artifact is missing: {partition_path}"
            )
        frame = pd.read_parquet(partition_path)
        positive_pairs = {
            (disease_map[str(row.source_id)], gene_map[str(row.target_id)])
            for row in frame.itertuples()
            if int(str(row.label)) == 1
        }
        overlap = message_pairs.intersection(positive_pairs)
        if overlap:
            raise ArtifactValidationError(
                f"{partition} supervision edges appear in the message graph"
            )

    try:
        model = load_training_checkpoint(
            paths.checkpoint,
            data.metadata(),
            training_config,
            expected_artifact_metadata={
                key: str(value) for key, value in artifact_metadata.items()
            },
        )
    except Exception as exc:
        raise ArtifactValidationError(
            f"Checkpoint compatibility validation failed: {exc}"
        ) from exc
    model.eval()
    model.is_trained = True

    return ValidatedArtifacts(
        paths=paths,
        data=data,
        model=model,
        training_config=training_config,
        model_config=model_config,
        dataset_manifest=dataset_manifest,
        graph_manifest=graph_manifest,
        feature_manifest=feature_manifest,
        node_index_map=node_index_map,
        split_metadata=split_metadata,
        validation_metrics=validation_metrics,
        run_manifest=run_manifest,
        checkpoint_sha256=sha256_file(paths.checkpoint),
        dataset_manifest_sha256=sha256_file(paths.dataset_manifest),
        artifact_metadata={key: str(value) for key, value in artifact_metadata.items()},
    )


def _attach_reference_node_metadata(
    data: HeteroData,
    graph_manifest: dict[str, Any],
    root: Path,
) -> None:
    node_records = [
        record
        for record in graph_manifest.get("outputs", [])
        if Path(str(record.get("path", ""))).name == "nodes.parquet"
    ]
    if not node_records:
        return
    nodes = pd.read_parquet(
        _resolve_record_path(str(node_records[0]["path"]), root)
    ).set_index("node_id", drop=False)
    type_map = {
        "Disease": "disease",
        "Drug": "drug",
        "Gene": "gene",
        "GOTerm": "go_term",
        "Pathway": "pathway",
    }
    for node_type in type_map.values():
        if node_type not in data.node_types:
            continue
        ids = [str(value) for value in data[node_type].node_id]
        for column in ("label", "symbol"):
            if column not in nodes.columns:
                continue
            values = [
                str(nodes.loc[node_id, column]) if node_id in nodes.index else ""
                for node_id in ids
            ]
            setattr(data[node_type], column, values)
