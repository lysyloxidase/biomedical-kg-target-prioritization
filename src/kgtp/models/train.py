"""Training loop for heterogeneous GNN link-prediction models."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike
from kgtp.eval.metrics import Query, Triple, evaluate_binary_and_ranking
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
    negatives_per_positive: int = 1_000
    edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES


@dataclass
class TrainingResult:
    """Model, loss curve, and filtered evaluation metrics for one seed."""

    model: MultiTaskLinkPredictor
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0


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
        )
    elif config.model_name == "graphsage":
        encoder = GraphSAGEEncoder(config.hidden_channels, config.num_layers, metadata)
    elif config.model_name == "graphsage_homogeneous":
        encoder = HomogeneousGraphSAGEEncoder(
            config.hidden_channels,
            config.num_layers,
            metadata[0],
        )
    elif config.model_name == "rgcn":
        encoder = RGCNEncoder(config.hidden_channels, config.num_layers, metadata)
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
    reference_data: HeteroData,
    *,
    seed: int,
    config: TrainingConfig,
    output_dir: PathLike | None = None,
) -> TrainingResult:
    """Train one seed and evaluate with the filtered Phase 3 protocol."""

    set_deterministic(seed)
    metadata = train_data.metadata()
    model = build_model(metadata, config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -math.inf
    stale_epochs = 0

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad()
        z_dict = model.encode(_x_dict(train_data), _edge_index_dict(train_data))
        loss = _supervision_loss(
            model, z_dict, train_data, criterion, config.edge_types
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        val_metrics = evaluate_split(
            model,
            val_data,
            reference_data,
            edge_types=config.edge_types,
            negatives_per_positive=config.negatives_per_positive,
            seed=seed,
        )
        score = val_metrics.get("AUPRC", 0.0) + val_metrics.get("filtered_MRR", 0.0)
        history.append(
            {"epoch": float(epoch), "loss": float(loss.item()), **val_metrics}
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_split(
        model,
        test_data,
        reference_data,
        edge_types=config.edge_types,
        negatives_per_positive=config.negatives_per_positive,
        seed=seed,
    )
    result = TrainingResult(
        model=model, metrics=test_metrics, history=history, best_epoch=best_epoch
    )
    if output_dir is not None:
        save_training_result(result, output_dir, config=config, seed=seed)
    return result


def evaluate_split(
    model: MultiTaskLinkPredictor,
    split_data: HeteroData,
    reference_data: HeteroData,
    *,
    edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES,
    negatives_per_positive: int = 1_000,
    seed: int = 13,
) -> dict[str, float]:
    """Evaluate a trained model with Phase 3 filtered metrics."""

    model.eval()
    with torch.no_grad():
        z_dict = model.encode(_x_dict(split_data), _edge_index_dict(split_data))

    positives: list[Triple] = []
    all_known: set[Triple] = set()
    tail_candidates: dict[Query, list[str]] = {}
    scorer = _model_scorer(model, z_dict, reference_data)

    for edge_type in edge_types:
        if edge_type not in split_data.edge_types or not hasattr(
            split_data[edge_type], "edge_label"
        ):
            continue
        positive_triples = _positive_label_triples(split_data, edge_type)
        positives.extend(positive_triples)
        all_known.update(_edge_index_triples(reference_data, edge_type))
        _, relation, dst_type = edge_type
        candidate_tails = _node_ids(reference_data, dst_type)
        for head, _, _ in positive_triples:
            tail_candidates[(head, relation)] = candidate_tails

    if not positives:
        return {"AUROC": math.nan, "AUPRC": math.nan, "filtered_MRR": math.nan}

    metrics = evaluate_binary_and_ranking(
        scorer,
        positives,
        all_known=all_known,
        tail_candidates=tail_candidates,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    filtered = metrics["filtered"]
    assert isinstance(filtered, dict)
    return {
        "AUROC": float(cast(float, metrics["AUROC"])),
        "AUPRC": float(cast(float, metrics["AUPRC"])),
        "filtered_MRR": float(filtered["MRR"]),
        "filtered_Hits@1": float(filtered["Hits@1"]),
        "filtered_Hits@3": float(filtered["Hits@3"]),
        "filtered_Hits@10": float(filtered["Hits@10"]),
    }


def train_multiseed(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    reference_data: HeteroData,
    *,
    config: TrainingConfig,
    seeds: tuple[int, ...] = (13, 17, 19, 23, 29),
    output_dir: PathLike = "reports/models",
    hgt_reference: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run the GNN training loop over at least five seeds and summarize."""

    if len(seeds) < 5:
        msg = "Phase 4 requires at least five seeds"
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
) -> None:
    """Persist model state, metrics, and history for reproducibility."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), output / "model.pt")
    payload = {
        "seed": seed,
        "config": config.__dict__,
        "metrics": result.metrics,
        "history": result.history,
        "best_epoch": result.best_epoch,
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
    torch.use_deterministic_algorithms(False)


def _supervision_loss(
    model: MultiTaskLinkPredictor,
    z_dict: dict[str, torch.Tensor],
    data: HeteroData,
    criterion: nn.Module,
    edge_types: tuple[EdgeType, ...],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for edge_type in edge_types:
        if edge_type not in data.edge_types or not hasattr(
            data[edge_type], "edge_label"
        ):
            continue
        logits = model.decode(z_dict, edge_type, data[edge_type].edge_label_index)
        labels = data[edge_type].edge_label.to(logits.device)
        losses.append(criterion(logits, labels))
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


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
        )
        with torch.no_grad():
            return float(model.decode(z_dict, edge_type, edge_label_index).item())

    return score


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
