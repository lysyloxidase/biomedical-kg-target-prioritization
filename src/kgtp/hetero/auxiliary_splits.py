"""Split optional auxiliary relations before train-graph feature fitting."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from kgtp.data.common import PathLike, ensure_columns, read_table, write_table
from kgtp.hetero.splits import EdgeType, SplitBundle
from kgtp.hetero.unlabeled_sampling import RandomUnlabeledSampler, SamplingContext


@dataclass(frozen=True)
class AuxiliaryTaskSpec:
    """Canonical table and PyG edge types for one auxiliary task."""

    name: str
    table: str
    source_node_type: str
    target_node_type: str
    edge_type: EdgeType
    reverse_edge_type: EdgeType


@dataclass(frozen=True)
class AuxiliaryTaskSplit:
    """Message and supervision partitions for one auxiliary relation."""

    spec: AuxiliaryTaskSpec
    full_known_positives: pd.DataFrame
    assignments: pd.DataFrame
    message_edges: pd.DataFrame
    train_supervision: pd.DataFrame
    validation_supervision: pd.DataFrame
    test_supervision: pd.DataFrame
    metadata: dict[str, Any]


AUXILIARY_TASK_SPECS: tuple[AuxiliaryTaskSpec, ...] = (
    AuxiliaryTaskSpec(
        name="drug_gene",
        table="drug_gene",
        source_node_type="Drug",
        target_node_type="Gene",
        edge_type=("drug", "targets", "gene"),
        reverse_edge_type=("gene", "rev_targets", "drug"),
    ),
    AuxiliaryTaskSpec(
        name="gene_pathway",
        table="gene_pathway",
        source_node_type="Gene",
        target_node_type="Pathway",
        edge_type=("gene", "participates_in", "pathway"),
        reverse_edge_type=("pathway", "rev_participates_in", "gene"),
    ),
)


def split_auxiliary_relations(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    seed: int,
) -> tuple[dict[str, AuxiliaryTaskSplit], dict[str, pd.DataFrame]]:
    """Create deterministic auxiliary supervision and train message tables."""

    train_tables = dict(edge_tables)
    splits: dict[str, AuxiliaryTaskSplit] = {}
    for offset, spec in enumerate(AUXILIARY_TASK_SPECS, start=1):
        split = _split_relation(
            nodes,
            edge_tables[spec.table],
            spec=spec,
            seed=seed + offset * 1_000,
        )
        splits[spec.name] = split
        train_tables[spec.table] = split.message_edges
    return splits, train_tables


def write_auxiliary_splits(
    splits: Mapping[str, AuxiliaryTaskSplit],
    output_dir: PathLike,
) -> list[Path]:
    """Persist auxiliary assignments, supervision, and task metadata."""

    outputs: list[Path] = []
    root = Path(output_dir) / "auxiliary"
    for name, split in sorted(splits.items()):
        task_dir = root / name
        frames = {
            "full_known_positives.parquet": split.full_known_positives,
            "assignments.parquet": split.assignments,
            "message_edges.parquet": split.message_edges,
            "supervision/train.parquet": split.train_supervision,
            "supervision/validation.parquet": split.validation_supervision,
            "supervision/test.parquet": split.test_supervision,
        }
        for relative, frame in frames.items():
            path = task_dir / relative
            write_table(frame, path)
            outputs.append(path)
        metadata_path = task_dir / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(split.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs.append(metadata_path)
    return outputs


def load_auxiliary_splits(input_dir: PathLike) -> dict[str, AuxiliaryTaskSplit]:
    """Load all configured auxiliary task splits."""

    root = Path(input_dir) / "auxiliary"
    splits: dict[str, AuxiliaryTaskSplit] = {}
    for spec in AUXILIARY_TASK_SPECS:
        task_dir = root / spec.name
        metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        splits[spec.name] = AuxiliaryTaskSplit(
            spec=spec,
            full_known_positives=read_table(task_dir / "full_known_positives.parquet"),
            assignments=read_table(task_dir / "assignments.parquet"),
            message_edges=read_table(task_dir / "message_edges.parquet"),
            train_supervision=read_table(task_dir / "supervision" / "train.parquet"),
            validation_supervision=read_table(
                task_dir / "supervision" / "validation.parquet"
            ),
            test_supervision=read_table(task_dir / "supervision" / "test.parquet"),
            metadata=metadata,
        )
    return splits


def attach_auxiliary_supervision(
    bundle: SplitBundle,
    splits: Mapping[str, AuxiliaryTaskSplit],
) -> SplitBundle:
    """Attach auxiliary labels to copies of the shared train message graph."""

    views = {
        "train": copy.deepcopy(bundle.train_data),
        "validation": copy.deepcopy(bundle.val_data),
        "test": copy.deepcopy(bundle.test_data),
    }
    for split in splits.values():
        for partition, supervision in (
            ("train", split.train_supervision),
            ("validation", split.validation_supervision),
            ("test", split.test_supervision),
        ):
            data = views[partition]
            edge_label_index, edge_label = _indexed_supervision(
                data,
                supervision,
                split.spec.edge_type,
            )
            data[split.spec.edge_type].edge_label_index = edge_label_index
            data[split.spec.edge_type].edge_label = edge_label
            _validate_message_exclusion(data, split, partition)
    metadata = dict(bundle.metadata)
    metadata["auxiliary_tasks"] = {
        name: split.metadata for name, split in sorted(splits.items())
    }
    return SplitBundle(
        train_data=views["train"],
        val_data=views["validation"],
        test_data=views["test"],
        metadata=metadata,
    )


def _split_relation(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    spec: AuxiliaryTaskSpec,
    seed: int,
) -> AuxiliaryTaskSplit:
    ensure_columns(
        edges,
        ("source_id", "target_id", "source_type", "edge_type", "target_type"),
    )
    ordered = edges.sort_values(["source_id", "target_id"]).reset_index(drop=True)
    if ordered.duplicated(["source_id", "target_id"]).any():
        msg = f"Auxiliary relation {spec.name} contains duplicate canonical pairs"
        raise ValueError(msg)
    if len(ordered) < 4:
        msg = f"Auxiliary relation {spec.name} requires at least four edges"
        raise ValueError(msg)

    indices = list(range(len(ordered)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(len(indices) * 0.2))
    validation_count = max(1, round((len(indices) - test_count) * 0.2))
    remaining = indices[test_count + validation_count :]
    train_count = max(1, round(len(remaining) * 0.2))
    if train_count >= len(remaining):
        train_count = len(remaining) - 1
    partition_indices = {
        "test": indices[:test_count],
        "validation": indices[test_count : test_count + validation_count],
        "train": remaining[:train_count],
        "message": remaining[train_count:],
    }
    assignments = pd.concat(
        [
            ordered.iloc[part][["source_id", "target_id"]].assign(partition=partition)
            for partition, part in partition_indices.items()
        ],
        ignore_index=True,
    ).sort_values(
        ["partition", "source_id", "target_id"],
        ignore_index=True,
    )
    known_pairs = _pairs(ordered)
    source_ids = tuple(
        sorted(
            nodes.loc[
                nodes["node_type"].astype(str) == spec.source_node_type,
                "node_id",
            ].astype(str)
        )
    )
    target_ids = tuple(
        sorted(
            nodes.loc[
                nodes["node_type"].astype(str) == spec.target_node_type,
                "node_id",
            ].astype(str)
        )
    )
    context = SamplingContext(
        source_ids=source_ids,
        target_ids=target_ids,
        known_positive_pairs=frozenset(known_pairs),
        target_properties={},
        hard_candidates_by_source={},
    )
    used_unlabeled: set[tuple[str, str]] = set()
    supervision: dict[str, pd.DataFrame] = {}
    for offset, partition in enumerate(("train", "validation", "test")):
        positives = sorted(_pairs(ordered.iloc[partition_indices[partition]]))
        unlabeled = RandomUnlabeledSampler().sample(
            context,
            positives,
            num_samples=len(positives),
            seed=seed + offset * 10_000,
            excluded_pairs=used_unlabeled,
        )
        used_unlabeled.update(unlabeled.pairs)
        supervision[partition] = _supervision_frame(positives, unlabeled.pairs)

    message_edges = ordered.iloc[partition_indices["message"]].sort_values(
        ["source_id", "target_id"],
        ignore_index=True,
    )
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "task": spec.name,
        "edge_type": list(spec.edge_type),
        "reverse_edge_type": list(spec.reverse_edge_type),
        "label_semantics": {
            "1": "positive",
            "0": "unlabeled_not_confirmed_negative",
        },
        "counts": {
            partition: len(partition_indices[partition])
            for partition in ("message", "train", "validation", "test")
        },
    }
    return AuxiliaryTaskSplit(
        spec=spec,
        full_known_positives=ordered[["source_id", "target_id"]].copy(),
        assignments=assignments,
        message_edges=message_edges,
        train_supervision=supervision["train"],
        validation_supervision=supervision["validation"],
        test_supervision=supervision["test"],
        metadata=metadata,
    )


def _supervision_frame(
    positives: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    unlabeled: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    rows = [
        {"source_id": source, "target_id": target, "label": 1}
        for source, target in positives
    ]
    rows.extend(
        {"source_id": source, "target_id": target, "label": 0}
        for source, target in unlabeled
    )
    return pd.DataFrame(rows).sort_values(
        ["label", "source_id", "target_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _indexed_supervision(
    data: Any,
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
    pairs = [
        (source_map[str(row.source_id)], target_map[str(row.target_id)])
        for row in supervision.itertuples(index=False)
    ]
    labels = [float(str(row.label)) for row in supervision.itertuples(index=False)]
    return (
        torch.tensor(pairs, dtype=torch.long).t().contiguous(),
        torch.tensor(labels, dtype=torch.float32),
    )


def _validate_message_exclusion(
    data: Any,
    split: AuxiliaryTaskSplit,
    partition: str,
) -> None:
    labels = data[split.spec.edge_type].edge_label
    positives = data[split.spec.edge_type].edge_label_index[:, labels == 1]
    message = {
        (int(source), int(target))
        for source, target in data[split.spec.edge_type].edge_index.t().tolist()
    }
    positive_pairs = {
        (int(source), int(target)) for source, target in positives.t().tolist()
    }
    if positive_pairs & message:
        msg = f"{split.spec.name} {partition} supervision overlaps message edges"
        raise ValueError(msg)
    reverse = {
        (int(source), int(target))
        for source, target in data[split.spec.reverse_edge_type].edge_index.t().tolist()
    }
    if {(target, source) for source, target in positive_pairs} & reverse:
        msg = (
            f"{split.spec.name} {partition} reverse supervision overlaps message edges"
        )
        raise ValueError(msg)


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(source), str(target))
        for source, target in zip(frame["source_id"], frame["target_id"], strict=True)
    }
