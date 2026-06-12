"""Training loop for heterogeneous GNN link-prediction models."""

from __future__ import annotations

import copy
import json
import math
import platform
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from kgtp import __version__
from kgtp.data.common import PathLike
from kgtp.eval.metrics import (
    Query,
    Triple,
    evaluate_full_candidate,
    evaluate_sampled_unlabeled,
)
from kgtp.eval.runner import paired_significance, summarize_seed_metrics
from kgtp.models.decoder import DecoderName
from kgtp.models.graphsage import GraphSAGEEncoder, HomogeneousGraphSAGEEncoder
from kgtp.models.hgt import HGTEncoder, Metadata
from kgtp.models.multitask import PREDICTED_EDGE_TYPES, MultiTaskLinkPredictor
from kgtp.models.rgcn import RGCNEncoder

ModelName = Literal["hgt", "graphsage", "graphsage_homogeneous", "rgcn"]
EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class TrainingConfig:
    """Small CPU-friendly training configuration."""

    model_name: ModelName = "hgt"
    hidden_channels: int = 32
    num_layers: int = 2
    num_heads: int = 4
    decoder_name: DecoderName = "dot"
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES
    disease_gene_weight: float = 1.0
    drug_gene_weight: float = 0.25
    gene_pathway_weight: float = 0.25
    dropout: float = 0.2
    residual: bool = True
    normalization: bool = True
    device: str = "auto"
    selection_metric: str = "primary_full_candidate_AUPRC"


@dataclass
class TrainingResult:
    """Model, loss curve, and filtered evaluation metrics for one seed."""

    model: MultiTaskLinkPredictor
    metrics: dict[str, float]
    task_metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    optimizer_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReference:
    """Known positives and candidate node IDs used only during evaluation."""

    known_triples: dict[EdgeType, set[Triple]]
    node_ids: dict[str, list[str]]


def build_model(
    metadata: Metadata,
    config: TrainingConfig,
) -> MultiTaskLinkPredictor:
    """Build HGT, GraphSAGE, homogeneous GraphSAGE, or R-GCN models."""

    encoder: nn.Module
    if config.model_name == "hgt":
        encoder = HGTEncoder(
            config.hidden_channels,
            config.num_heads,
            config.num_layers,
            metadata,
            dropout=config.dropout,
            residual=config.residual,
            normalization=config.normalization,
        )
    elif config.model_name == "graphsage":
        encoder = GraphSAGEEncoder(
            config.hidden_channels,
            config.num_layers,
            metadata,
            dropout=config.dropout,
            residual=config.residual,
            normalization=config.normalization,
        )
    elif config.model_name == "graphsage_homogeneous":
        encoder = HomogeneousGraphSAGEEncoder(
            config.hidden_channels,
            config.num_layers,
            metadata[0],
            dropout=config.dropout,
            residual=config.residual,
            normalization=config.normalization,
        )
    elif config.model_name == "rgcn":
        encoder = RGCNEncoder(
            config.hidden_channels,
            config.num_layers,
            metadata,
            dropout=config.dropout,
            residual=config.residual,
            normalization=config.normalization,
        )
    else:
        msg = f"Unsupported model: {config.model_name}"
        raise ValueError(msg)

    return MultiTaskLinkPredictor(
        encoder,
        hidden_channels=config.hidden_channels,
        edge_types=config.edge_types,
        decoder_name=config.decoder_name,
    )


def train_one_seed(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    reference_data: HeteroData | EvaluationReference,
    *,
    seed: int,
    config: TrainingConfig,
    output_dir: PathLike | None = None,
    artifact_metadata: dict[str, str] | None = None,
) -> TrainingResult:
    """Train one seed with weighted tasks and best-validation checkpointing."""

    set_deterministic(seed)
    device = _resolve_device(config.device)
    train_view = copy.deepcopy(train_data).to(device)
    val_view = copy.deepcopy(val_data).to(device)
    test_view = copy.deepcopy(test_data).to(device)
    metadata = train_data.metadata()
    model = build_model(metadata, config).to(device)
    with torch.no_grad():
        model.encode(_x_dict(train_view), _edge_index_dict(train_view))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_epoch = 0
    best_score = -math.inf
    stale_epochs = 0

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad()
        z_dict = model.encode(_x_dict(train_view), _edge_index_dict(train_view))
        loss, task_losses = _supervision_loss(
            model,
            z_dict,
            train_view,
            criterion,
            config,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        val_detailed = evaluate_split_detailed(
            model,
            val_view,
            reference_data,
            edge_types=config.edge_types,
            require_trained=False,
        )
        val_metrics = _primary_flat_metrics(val_detailed)
        score = val_metrics.get("AUPRC", -math.inf)
        history.append(
            {
                "epoch": float(epoch),
                "total_loss": float(loss.item()),
                **{
                    f"loss_{task}": value for task, value in sorted(task_losses.items())
                },
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.is_trained = True
    test_detailed = evaluate_split_detailed(
        model,
        test_view,
        reference_data,
        edge_types=config.edge_types,
    )
    test_metrics = _primary_flat_metrics(test_detailed)
    model = model.cpu()
    result = TrainingResult(
        model=model,
        metrics=test_metrics,
        task_metrics=test_detailed,
        history=history,
        best_epoch=best_epoch,
        optimizer_state=best_optimizer_state or optimizer.state_dict(),
    )
    if output_dir is not None:
        save_training_result(
            result,
            output_dir,
            config=config,
            seed=seed,
            artifact_metadata=artifact_metadata or {},
            device=str(device),
        )
    return result


def evaluate_split(
    model: MultiTaskLinkPredictor,
    split_data: HeteroData,
    reference_data: HeteroData | EvaluationReference,
    *,
    edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES,
    negatives_per_positive: int = 1_000,
    seed: int = 13,
) -> dict[str, float]:
    """Return flattened primary-task full-candidate metrics."""

    del negatives_per_positive, seed
    return _primary_flat_metrics(
        evaluate_split_detailed(
            model,
            split_data,
            reference_data,
            edge_types=edge_types,
        )
    )


def evaluate_split_detailed(
    model: MultiTaskLinkPredictor,
    split_data: HeteroData,
    reference_data: HeteroData | EvaluationReference,
    *,
    edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES,
    sampled_unlabeled: dict[str, list[Triple]] | None = None,
    require_trained: bool = True,
) -> dict[str, Any]:
    """Evaluate every task with full candidates and explicit unlabeled sets."""

    if require_trained and not model.is_trained:
        msg = "Refusing to evaluate an untrained GNN model"
        raise RuntimeError(msg)

    model.eval()
    with torch.no_grad():
        z_dict = model.encode(_x_dict(split_data), _edge_index_dict(split_data))

    reference = _as_evaluation_reference(reference_data, edge_types=edge_types)
    scorer = _model_scorer(model, z_dict, split_data)

    def probability_scorer(triple: Triple) -> float:
        return float(torch.sigmoid(torch.tensor(scorer(triple))))

    task_metrics: dict[str, Any] = {}
    for edge_type in edge_types:
        if edge_type not in split_data.edge_types or not hasattr(
            split_data[edge_type], "edge_label"
        ):
            continue
        positives = _positive_label_triples(split_data, edge_type)
        unlabeled = _unlabeled_label_triples(split_data, edge_type)
        all_known = reference.known_triples.get(edge_type, set())
        _, relation, dst_type = edge_type
        candidate_tails = reference.node_ids[dst_type]
        tail_candidates: dict[Query, list[str]] = {
            (head, relation): candidate_tails for head, _, _ in positives
        }
        if not positives:
            continue
        key = _edge_key(edge_type)
        sampled = {
            "split_random": evaluate_sampled_unlabeled(
                scorer,
                positives,
                unlabeled,
                strategy="split_random",
                probability_scorer=probability_scorer,
            )
        }
        if edge_type == PREDICTED_EDGE_TYPES[0] and sampled_unlabeled is not None:
            for strategy, triples in sorted(sampled_unlabeled.items()):
                sampled[strategy] = evaluate_sampled_unlabeled(
                    scorer,
                    positives,
                    triples,
                    strategy=strategy,
                    probability_scorer=probability_scorer,
                )
        task_metrics[key] = {
            "edge_type": list(edge_type),
            "full_candidate": evaluate_full_candidate(
                scorer,
                positives,
                all_known=all_known,
                tail_candidates=tail_candidates,
                probability_scorer=probability_scorer,
            ),
            "sampled_unlabeled": sampled,
        }
    return {
        "primary_task": _edge_key(PREDICTED_EDGE_TYPES[0]),
        "tasks": task_metrics,
    }


def train_multiseed(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    reference_data: HeteroData | EvaluationReference,
    *,
    config: TrainingConfig,
    seeds: tuple[int, ...] = (13, 17, 19, 23, 29),
    output_dir: PathLike = "reports/models",
    hgt_reference: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run the GNN training loop over at least five seeds and summarize."""

    if len(seeds) < 5:
        msg = "Phase 5 requires at least five seeds"
        raise ValueError(msg)
    seed_results: dict[str, dict[str, float]] = {}
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = train_one_seed(
            train_data,
            val_data,
            test_data,
            reference_data,
            seed=seed,
            config=config,
            output_dir=model_dir / f"{config.model_name}_{seed}",
        )
        seed_results[str(seed)] = result.metrics
    payload = {
        "model": config.model_name,
        "primary_metric": "AUPRC",
        "seeds": list(seeds),
        "seed_results": seed_results,
        "summary": summarize_seed_metrics(seed_results),
        "paired_significance_vs_hgt": paired_significance(seed_results, hgt_reference),
    }
    (model_dir / f"results_{config.model_name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def hgt_beats_popularity_floor(
    hgt_metrics: dict[str, float],
    popularity_metrics: dict[str, float],
    *,
    metric: str = "AUPRC",
) -> bool:
    """Sanity gate: HGT should clear the popularity floor."""

    return hgt_metrics[metric] > popularity_metrics[metric]


def save_training_result(
    result: TrainingResult,
    output_dir: PathLike,
    *,
    config: TrainingConfig,
    seed: int,
    artifact_metadata: dict[str, str] | None = None,
    device: str = "cpu",
) -> None:
    """Persist model state, metrics, and history for reproducibility."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), output / "model.pt")
    checkpoint = {
        "schema_version": 1,
        "model_state": result.model.state_dict(),
        "optimizer_state": result.optimizer_state,
        "config": config.__dict__,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "artifact_metadata": artifact_metadata or {},
    }
    torch.save(checkpoint, output / "best_checkpoint.pt")
    config_payload = {
        "schema_version": 1,
        "model_name": config.model_name,
        "model_version": __version__,
        "trained": True,
        "config": config.__dict__,
        "artifact_metadata": artifact_metadata or {},
        "checkpoint": "best_checkpoint.pt",
        "selection_metric": config.selection_metric,
    }
    (output / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "seed": seed,
        "config": config.__dict__,
        "metrics": result.metrics,
        "task_metrics": result.task_metrics,
        "history": result.history,
        "best_epoch": result.best_epoch,
        "checkpoint": "best_checkpoint.pt",
        "device": device,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "full_batch": True,
            "limitation": (
                "Full-batch training is appropriate only for the small sample; "
                "production graphs require neighbor sampling or partitioning."
            ),
        },
        "artifact_metadata": artifact_metadata or {},
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def set_deterministic(seed: int) -> None:
    """Set seeds for deterministic small-graph experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _supervision_loss(
    model: MultiTaskLinkPredictor,
    z_dict: dict[str, torch.Tensor],
    data: HeteroData,
    criterion: nn.Module,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    weighted_losses: list[torch.Tensor] = []
    task_losses: dict[str, float] = {}
    for edge_type in config.edge_types:
        if edge_type not in data.edge_types or not hasattr(
            data[edge_type], "edge_label"
        ):
            continue
        logits = model.decode(z_dict, edge_type, data[edge_type].edge_label_index)
        labels = data[edge_type].edge_label.to(logits.device)
        task_loss = criterion(logits, labels)
        weight = _task_weight(config, edge_type)
        task_losses[_edge_key(edge_type)] = float(task_loss.detach().item())
        if weight > 0:
            weighted_losses.append(task_loss * weight)
    if not weighted_losses:
        msg = "No positive-weight supervision task is available for training"
        raise ValueError(msg)
    return torch.stack(weighted_losses).sum(), task_losses


def _model_scorer(
    model: MultiTaskLinkPredictor,
    z_dict: dict[str, torch.Tensor],
    data: HeteroData,
) -> Callable[[Triple], float]:
    node_maps = {
        node_type: {
            node_id: index for index, node_id in enumerate(_node_ids(data, node_type))
        }
        for node_type in data.node_types
    }
    relation_to_edge_type = {
        relation: edge_type
        for edge_type in model.edge_types
        for relation in (edge_type[1],)
    }

    def score(triple: Triple) -> float:
        head, relation, tail = triple
        edge_type = relation_to_edge_type[relation]
        src_type, _, dst_type = edge_type
        edge_label_index = torch.tensor(
            [[node_maps[src_type][head]], [node_maps[dst_type][tail]]],
            dtype=torch.long,
            device=z_dict[src_type].device,
        )
        with torch.no_grad():
            return float(model.decode(z_dict, edge_type, edge_label_index).item())

    return score


def _as_evaluation_reference(
    reference: HeteroData | EvaluationReference,
    *,
    edge_types: tuple[EdgeType, ...],
) -> EvaluationReference:
    if isinstance(reference, EvaluationReference):
        return reference
    return EvaluationReference(
        known_triples={
            edge_type: set(_edge_index_triples(reference, edge_type))
            for edge_type in edge_types
        },
        node_ids={
            node_type: _node_ids(reference, node_type)
            for node_type in reference.node_types
        },
    )


def _positive_label_triples(data: HeteroData, edge_type: EdgeType) -> list[Triple]:
    src_type, relation, dst_type = edge_type
    labels = data[edge_type].edge_label
    edge_label_index = data[edge_type].edge_label_index
    src_ids = _node_ids(data, src_type)
    dst_ids = _node_ids(data, dst_type)
    triples: list[Triple] = []
    for source, target, label in zip(
        edge_label_index[0].tolist(),
        edge_label_index[1].tolist(),
        labels.tolist(),
        strict=True,
    ):
        if float(label) == 1.0:
            triples.append((src_ids[int(source)], relation, dst_ids[int(target)]))
    return triples


def _unlabeled_label_triples(data: HeteroData, edge_type: EdgeType) -> list[Triple]:
    src_type, relation, dst_type = edge_type
    labels = data[edge_type].edge_label
    edge_label_index = data[edge_type].edge_label_index
    src_ids = _node_ids(data, src_type)
    dst_ids = _node_ids(data, dst_type)
    return [
        (src_ids[int(source)], relation, dst_ids[int(target)])
        for source, target, label in zip(
            edge_label_index[0].tolist(),
            edge_label_index[1].tolist(),
            labels.tolist(),
            strict=True,
        )
        if float(label) == 0.0
    ]


def _edge_index_triples(data: HeteroData, edge_type: EdgeType) -> list[Triple]:
    src_type, relation, dst_type = edge_type
    if edge_type not in data.edge_types:
        return []
    src_ids = _node_ids(data, src_type)
    dst_ids = _node_ids(data, dst_type)
    edge_index = data[edge_type].edge_index
    return [
        (src_ids[int(source)], relation, dst_ids[int(target)])
        for source, target in edge_index.t().tolist()
    ]


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]


def _x_dict(data: HeteroData) -> dict[str, torch.Tensor]:
    return {node_type: data[node_type].x for node_type in data.node_types}


def _edge_index_dict(data: HeteroData) -> dict[EdgeType, torch.Tensor]:
    return {
        edge_type: data[edge_type].edge_index
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }


def load_training_checkpoint(
    path: PathLike,
    metadata: Metadata,
    config: TrainingConfig,
    *,
    expected_artifact_metadata: dict[str, str],
) -> MultiTaskLinkPredictor:
    """Load a checkpoint only when model and artifact manifests are compatible."""

    payload = torch.load(Path(path), weights_only=False, map_location="cpu")
    if payload.get("config") != config.__dict__:
        msg = "Checkpoint training configuration is incompatible"
        raise ValueError(msg)
    if payload.get("artifact_metadata") != expected_artifact_metadata:
        msg = "Checkpoint dataset/split metadata is incompatible"
        raise ValueError(msg)
    model = build_model(metadata, config)
    model.load_state_dict(payload["model_state"])
    model.is_trained = True
    return model


def _task_weight(config: TrainingConfig, edge_type: EdgeType) -> float:
    weights = {
        PREDICTED_EDGE_TYPES[0]: config.disease_gene_weight,
        PREDICTED_EDGE_TYPES[1]: config.drug_gene_weight,
        PREDICTED_EDGE_TYPES[2]: config.gene_pathway_weight,
    }
    return weights.get(edge_type, 0.0)


def _primary_flat_metrics(detailed: dict[str, Any]) -> dict[str, float]:
    task = detailed["tasks"].get(detailed["primary_task"])
    if task is None:
        return {"AUROC": math.nan, "AUPRC": math.nan, "filtered_MRR": math.nan}
    full = cast(dict[str, Any], task["full_candidate"])
    filtered = cast(dict[str, float], full["filtered"])
    return {
        "AUROC": float(full["AUROC"]),
        "AUPRC": float(full["AUPRC"]),
        "filtered_MRR": float(filtered["MRR"]),
        "filtered_Hits@1": float(filtered["Hits@1"]),
        "filtered_Hits@3": float(filtered["Hits@3"]),
        "filtered_Hits@10": float(filtered["Hits@10"]),
        "filtered_Hits@50": float(filtered["Hits@50"]),
    }


def _edge_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is unavailable"
        raise RuntimeError(msg)
    return device
