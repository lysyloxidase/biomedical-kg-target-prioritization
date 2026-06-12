"""Split-first protocol for canonical target-relation tables."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike, ensure_columns, read_table, write_table
from kgtp.hetero.splits import EdgeType, SplitBundle
from kgtp.hetero.unlabeled_sampling import (
    RandomUnlabeledSampler,
    SamplingContext,
)

TARGET_TABLE = "disease_gene"
TARGET_COLUMNS = ("source_id", "target_id")
PARTITIONS = ("message", "train", "validation", "test")


@dataclass(frozen=True)
class TargetSplit:
    """Canonical target-edge partitions and supervision tables."""

    full_known_positives: pd.DataFrame
    assignments: pd.DataFrame
    message_edges: pd.DataFrame
    train_supervision: pd.DataFrame
    validation_supervision: pd.DataFrame
    test_supervision: pd.DataFrame
    metadata: dict[str, Any]


def split_target_relation(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    seed: int = 13,
    num_val: float | int = 0.2,
    num_test: float | int = 0.2,
    disjoint_train_ratio: float | int = 0.3,
) -> TargetSplit:
    """Partition canonical target edges before any graph-derived features."""

    target_edges = edge_tables.get(TARGET_TABLE)
    if target_edges is None:
        msg = f"Missing target relation table: {TARGET_TABLE}"
        raise ValueError(msg)
    _validate_target_table(target_edges)
    ordered = target_edges.sort_values(list(TARGET_COLUMNS)).reset_index(drop=True)
    edge_count = len(ordered)
    if edge_count < 4:
        msg = "Target relation requires at least four positive edges"
        raise ValueError(msg)

    permutation = list(range(edge_count))
    random.Random(seed).shuffle(permutation)
    test_count = _count_from_ratio(num_test, edge_count)
    val_count = _count_from_ratio(num_val, edge_count - test_count)
    train_indices = permutation[test_count + val_count :]
    train_supervision_count = _count_from_ratio(
        disjoint_train_ratio, len(train_indices)
    )
    if train_supervision_count == 0 and train_indices:
        train_supervision_count = 1
    if train_supervision_count >= len(train_indices):
        train_supervision_count = len(train_indices) - 1
    if min(test_count, val_count, train_supervision_count) <= 0:
        msg = "Split parameters produced an empty positive supervision partition"
        raise ValueError(msg)

    partition_indices = {
        "test": permutation[:test_count],
        "validation": permutation[test_count : test_count + val_count],
        "train": train_indices[:train_supervision_count],
        "message": train_indices[train_supervision_count:],
    }
    assignments = pd.concat(
        [
            ordered.iloc[indices][list(TARGET_COLUMNS)].assign(partition=partition)
            for partition, indices in partition_indices.items()
        ],
        ignore_index=True,
    ).sort_values(["partition", *TARGET_COLUMNS], ignore_index=True)
    full_known = ordered[list(TARGET_COLUMNS)].copy()
    message_edges = ordered.iloc[partition_indices["message"]].sort_values(
        list(TARGET_COLUMNS), ignore_index=True
    )

    known_pairs = _pairs(full_known)
    source_ids = sorted(full_known["source_id"].astype(str).unique())
    target_ids = sorted(
        nodes.loc[nodes["node_type"] == "Gene", "node_id"].astype(str).unique()
    )
    used_negatives: set[tuple[str, str]] = set()
    supervision: dict[str, pd.DataFrame] = {}
    for offset, partition in enumerate(("train", "validation", "test")):
        positive_rows = ordered.iloc[partition_indices[partition]]
        positive_pairs = sorted(_pairs(positive_rows))
        negatives = list(
            RandomUnlabeledSampler()
            .sample(
                SamplingContext(
                    source_ids=tuple(source_ids),
                    target_ids=tuple(target_ids),
                    known_positive_pairs=frozenset(known_pairs),
                    target_properties={},
                    hard_candidates_by_source={},
                ),
                positive_pairs,
                num_samples=len(positive_pairs),
                seed=seed + offset * 10_000,
                excluded_pairs=used_negatives,
            )
            .pairs
        )
        used_negatives.update(negatives)
        supervision[partition] = _supervision_frame(positive_pairs, negatives)

    node_index_hash = hash_node_index_maps(nodes)
    reference_graph_hash = hash_graph(nodes, edge_tables)
    train_edge_tables = dict(edge_tables)
    train_edge_tables[TARGET_TABLE] = message_edges
    train_graph_hash = hash_graph(nodes, train_edge_tables)
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "target_relation": ["Disease", "associated_with", "Gene"],
        "num_val": num_val,
        "num_test": num_test,
        "disjoint_train_ratio": disjoint_train_ratio,
        "relation_policy": {
            "disease_gene": "partitioned before feature fitting",
            "non_target_relations": "full transductive context shared across splits",
        },
        "full_reference_graph_hash": reference_graph_hash,
        "train_message_graph_hash": train_graph_hash,
        "node_index_map_hash": node_index_hash,
        "partition_hashes": {
            partition: hash_frame(
                assignments.loc[
                    assignments["partition"] == partition,
                    [*TARGET_COLUMNS, "partition"],
                ]
            )
            for partition in PARTITIONS
        },
        "counts": {
            partition: len(partition_indices[partition]) for partition in PARTITIONS
        },
    }
    split = TargetSplit(
        full_known_positives=full_known,
        assignments=assignments,
        message_edges=message_edges,
        train_supervision=supervision["train"],
        validation_supervision=supervision["validation"],
        test_supervision=supervision["test"],
        metadata=metadata,
    )
    validate_target_split(split, nodes=nodes, edge_tables=edge_tables)
    return split


def validate_target_split(
    split: TargetSplit,
    *,
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
    train_edge_tables: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    """Reject incomplete, overlapping, duplicated, or hash-incompatible splits."""

    target_edges = edge_tables.get(TARGET_TABLE)
    if target_edges is None:
        msg = f"Missing target relation table: {TARGET_TABLE}"
        raise ValueError(msg)
    _validate_target_table(target_edges)
    reference = _pairs(target_edges)
    if reference != _pairs(split.full_known_positives):
        msg = "Full known-positive registry does not match the reference graph"
        raise ValueError(msg)

    ensure_columns(split.assignments, (*TARGET_COLUMNS, "partition"))
    invalid_partitions = set(split.assignments["partition"]) - set(PARTITIONS)
    if invalid_partitions:
        msg = f"Unknown split partitions: {sorted(invalid_partitions)}"
        raise ValueError(msg)
    if split.assignments.duplicated(list(TARGET_COLUMNS)).any():
        msg = "A canonical target edge is assigned to more than one partition"
        raise ValueError(msg)

    assigned: dict[str, set[tuple[str, str]]] = {
        partition: _pairs(
            split.assignments.loc[split.assignments["partition"] == partition]
        )
        for partition in PARTITIONS
    }
    _assert_pairwise_disjoint(assigned)
    reconstructed = set().union(*assigned.values())
    if reconstructed != reference:
        msg = "Target partitions do not reconstruct the full positive registry"
        raise ValueError(msg)
    if assigned["message"] != _pairs(split.message_edges):
        msg = "Train message edges do not match message assignments"
        raise ValueError(msg)

    supervision = {
        "train": split.train_supervision,
        "validation": split.validation_supervision,
        "test": split.test_supervision,
    }
    negative_sets: dict[str, set[tuple[str, str]]] = {}
    for partition, frame in supervision.items():
        ensure_columns(frame, (*TARGET_COLUMNS, "label"))
        if set(frame["label"].astype(int).unique()) != {0, 1}:
            msg = f"{partition} supervision must contain labels 0 and 1"
            raise ValueError(msg)
        if frame.duplicated(list(TARGET_COLUMNS)).any():
            msg = f"{partition} supervision contains duplicate canonical edges"
            raise ValueError(msg)
        positives = _pairs(frame.loc[frame["label"] == 1])
        if positives != assigned[partition]:
            msg = f"{partition} positive supervision does not match assignments"
            raise ValueError(msg)
        negatives = _pairs(frame.loc[frame["label"] == 0])
        if negatives & reference:
            msg = f"{partition} negatives overlap known positive edges"
            raise ValueError(msg)
        negative_sets[partition] = negatives
    _assert_pairwise_disjoint(negative_sets)

    if assigned["message"] & assigned["train"]:
        msg = "Train supervision edges are present in the message graph"
        raise ValueError(msg)
    for partition in ("validation", "test"):
        if assigned["message"] & assigned[partition]:
            msg = f"{partition} edges are present in the message graph"
            raise ValueError(msg)

    expected_reference_hash = hash_graph(nodes, edge_tables)
    expected_train_tables = (
        dict(train_edge_tables) if train_edge_tables is not None else dict(edge_tables)
    )
    expected_train_tables[TARGET_TABLE] = split.message_edges
    expected_train_hash = hash_graph(nodes, expected_train_tables)
    expected_node_hash = hash_node_index_maps(nodes)
    if split.metadata.get("full_reference_graph_hash") != expected_reference_hash:
        msg = "Full reference graph hash does not match split metadata"
        raise ValueError(msg)
    if split.metadata.get("train_message_graph_hash") != expected_train_hash:
        msg = "Train message graph hash does not match split metadata"
        raise ValueError(msg)
    if split.metadata.get("node_index_map_hash") != expected_node_hash:
        msg = "Node-index map hash does not match split metadata"
        raise ValueError(msg)
    for partition in PARTITIONS:
        expected = hash_frame(
            split.assignments.loc[
                split.assignments["partition"] == partition,
                [*TARGET_COLUMNS, "partition"],
            ]
        )
        recorded = split.metadata.get("partition_hashes", {}).get(partition)
        if recorded != expected:
            msg = f"{partition} partition hash does not match split metadata"
            raise ValueError(msg)


def write_target_split(split: TargetSplit, output_dir: PathLike) -> list[Path]:
    """Write split registry, assignments, supervision, and metadata."""

    output = Path(output_dir)
    supervision_dir = output / "supervision"
    paths = [
        output / "full_known_positives.parquet",
        output / "split_assignments.parquet",
        output / "message_edges.parquet",
        supervision_dir / "train.parquet",
        supervision_dir / "validation.parquet",
        supervision_dir / "test.parquet",
    ]
    frames = [
        split.full_known_positives,
        split.assignments,
        split.message_edges,
        split.train_supervision,
        split.validation_supervision,
        split.test_supervision,
    ]
    for path, frame in zip(paths, frames, strict=True):
        write_table(frame, path)
    metadata_path = output / "split_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(split.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [*paths, metadata_path]


def load_target_split(input_dir: PathLike) -> TargetSplit:
    """Load canonical split artifacts from disk."""

    root = Path(input_dir)
    metadata = json.loads((root / "split_metadata.json").read_text(encoding="utf-8"))
    return TargetSplit(
        full_known_positives=read_table(root / "full_known_positives.parquet"),
        assignments=read_table(root / "split_assignments.parquet"),
        message_edges=read_table(root / "message_edges.parquet"),
        train_supervision=read_table(root / "supervision" / "train.parquet"),
        validation_supervision=read_table(root / "supervision" / "validation.parquet"),
        test_supervision=read_table(root / "supervision" / "test.parquet"),
        metadata=metadata,
    )


def build_split_bundle(
    message_data: HeteroData,
    split: TargetSplit,
    *,
    edge_type: EdgeType,
    reverse_edge_type: EdgeType,
) -> SplitBundle:
    """Attach supervision to three views of one train-only message graph."""

    bundle_data: dict[str, HeteroData] = {}
    for split_name, supervision in (
        ("train", split.train_supervision),
        ("validation", split.validation_supervision),
        ("test", split.test_supervision),
    ):
        data = copy.deepcopy(message_data)
        edge_label_index, edge_label = _indexed_supervision(
            data, supervision, edge_type
        )
        data[edge_type].edge_label_index = edge_label_index
        data[edge_type].edge_label = edge_label
        data.split_name = split_name
        data.split_seed = int(split.metadata["seed"])
        bundle_data[split_name] = data
    bundle = SplitBundle(
        train_data=bundle_data["train"],
        val_data=bundle_data["validation"],
        test_data=bundle_data["test"],
        metadata=split.metadata,
    )
    validate_split_bundle_graphs(
        bundle,
        edge_type=edge_type,
        reverse_edge_type=reverse_edge_type,
    )
    return bundle


def validate_split_bundle_graphs(
    bundle: SplitBundle,
    *,
    edge_type: EdgeType,
    reverse_edge_type: EdgeType,
) -> None:
    """Ensure every model view has the same train-only message graph."""

    message_pairs = _tensor_pairs(bundle.train_data[edge_type].edge_index)
    reverse_message_pairs = _tensor_pairs(
        bundle.train_data[reverse_edge_type].edge_index
    )
    if reverse_message_pairs != {(target, source) for source, target in message_pairs}:
        msg = "Reverse target edges do not mirror the train message graph"
        raise ValueError(msg)
    feature_reference = {
        node_type: bundle.train_data[node_type].x
        for node_type in bundle.train_data.node_types
    }
    for name, data in (
        ("validation", bundle.val_data),
        ("test", bundle.test_data),
    ):
        if _tensor_pairs(data[edge_type].edge_index) != message_pairs:
            msg = f"{name} view does not use the train message graph"
            raise ValueError(msg)
        if _tensor_pairs(data[reverse_edge_type].edge_index) != reverse_message_pairs:
            msg = f"{name} view contains incompatible reverse target edges"
            raise ValueError(msg)
        for node_type, expected in feature_reference.items():
            if not torch.equal(data[node_type].x, expected):
                msg = f"{name} view has different {node_type} features"
                raise ValueError(msg)
    for name, data in (
        ("train", bundle.train_data),
        ("validation", bundle.val_data),
        ("test", bundle.test_data),
    ):
        labels = data[edge_type].edge_label
        positives = _tensor_pairs(data[edge_type].edge_label_index[:, labels == 1])
        if positives & message_pairs:
            msg = f"{name} positive supervision overlaps message edges"
            raise ValueError(msg)
        reverse_positives = {(target, source) for source, target in positives}
        if reverse_positives & reverse_message_pairs:
            msg = f"{name} reverse supervision overlaps reverse message edges"
            raise ValueError(msg)


def hash_graph(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
) -> str:
    """Hash canonical graph content independent of Parquet encoding."""

    digest = hashlib.sha256()
    digest.update(hash_frame(nodes).encode())
    for name, frame in sorted(edge_tables.items()):
        digest.update(name.encode())
        digest.update(hash_frame(frame).encode())
    return digest.hexdigest()


def hash_node_index_maps(nodes: pd.DataFrame) -> str:
    """Hash deterministic node-type to node-index assignments."""

    maps = {
        _canonical_node_type(str(node_type)): {
            node_id: index
            for index, node_id in enumerate(
                sorted(group["node_id"].astype(str).unique())
            )
        }
        for node_type, group in nodes.groupby("node_type", sort=True)
    }
    encoded = json.dumps(maps, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_node_type(node_type: str) -> str:
    return {
        "Disease": "disease",
        "Drug": "drug",
        "Gene": "gene",
        "GOTerm": "go_term",
        "Pathway": "pathway",
    }.get(node_type, node_type.lower())


def hash_frame(frame: pd.DataFrame) -> str:
    """Hash a dataframe after deterministic row and column ordering."""

    columns = sorted(str(column) for column in frame.columns)
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    normalized = normalized[columns].fillna("")
    if columns:
        normalized = normalized.sort_values(columns, kind="mergesort")
    payload = normalized.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_target_table(target_edges: pd.DataFrame) -> None:
    ensure_columns(
        target_edges,
        (*TARGET_COLUMNS, "source_type", "edge_type", "target_type"),
    )
    if target_edges.duplicated(list(TARGET_COLUMNS)).any():
        msg = "Target relation contains duplicate canonical edges"
        raise ValueError(msg)
    triples = set(
        zip(
            target_edges["source_type"],
            target_edges["edge_type"],
            target_edges["target_type"],
            strict=True,
        )
    )
    if triples != {("Disease", "associated_with", "Gene")}:
        msg = f"Unexpected target relation schema: {sorted(triples)}"
        raise ValueError(msg)


def _supervision_frame(
    positives: Sequence[tuple[str, str]],
    negatives: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows = [
        {"source_id": source, "target_id": target, "label": 1}
        for source, target in positives
    ]
    rows.extend(
        {"source_id": source, "target_id": target, "label": 0}
        for source, target in negatives
    )
    return pd.DataFrame(rows).sort_values(
        ["label", *TARGET_COLUMNS],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _indexed_supervision(
    data: HeteroData,
    supervision: pd.DataFrame,
    edge_type: EdgeType,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_type, _, target_type = edge_type
    source_map = {
        str(node_id): index for index, node_id in enumerate(data[source_type].node_id)
    }
    target_map = {
        str(node_id): index for index, node_id in enumerate(data[target_type].node_id)
    }
    pairs: list[tuple[int, int]] = []
    labels: list[float] = []
    for row in supervision.itertuples(index=False):
        if str(row.source_id) not in source_map or str(row.target_id) not in target_map:
            msg = f"Supervision endpoint is missing from node maps: {row}"
            raise ValueError(msg)
        pairs.append(
            (
                source_map[str(row.source_id)],
                target_map[str(row.target_id)],
            )
        )
        labels.append(float(str(row.label)))
    return (
        torch.tensor(pairs, dtype=torch.long).t().contiguous(),
        torch.tensor(labels, dtype=torch.float32),
    )


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    ensure_columns(frame, TARGET_COLUMNS)
    return {
        (str(source), str(target))
        for source, target in zip(frame["source_id"], frame["target_id"], strict=True)
    }


def _tensor_pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(source), int(target)) for source, target in edge_index.t().tolist()}


def _assert_pairwise_disjoint(
    partitions: Mapping[str, set[tuple[str, str]]],
) -> None:
    names = sorted(partitions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if partitions[first] & partitions[second]:
                msg = f"Split partitions overlap: {first} and {second}"
                raise ValueError(msg)


def _count_from_ratio(value: float | int, total: int) -> int:
    if total <= 0:
        return 0
    if isinstance(value, int):
        return max(0, min(value, total))
    return max(0, min(round(total * value), total))
