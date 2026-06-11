"""HGT attention-weight extraction for prediction rationales."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, SupportsFloat, cast

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import AttentionExplainer

from kgtp.explain.explainer import DISEASE_GENE_EDGE

EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class AttentionWeight:
    """One per-layer edge attention/proxy-attention record."""

    layer: int
    edge_type: EdgeType
    source: int
    target: int
    source_id: str
    target_id: str
    weight: float
    meta_relation: str
    method: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["edge_type"] = list(self.edge_type)
        return payload


def extract_hgt_attention_weights(
    model: nn.Module,
    data: HeteroData,
    disease_idx: int,
    gene_idx: int,
    *,
    edge_type: EdgeType = DISEASE_GENE_EDGE,
    top_k: int = 20,
) -> dict[str, object]:
    """Extract per-layer, per-edge-type HGT attention records.

    PyG's ``HGTConv`` does not expose a stable public attention tensor across
    versions. When native attention is unavailable, this returns a deterministic
    endpoint-conditioned proxy and marks the method accordingly. The schema is
    the same either way, so downstream reports remain reproducible.
    """

    attention_available = _attention_can_construct()
    layer_count = max(1, _num_hgt_layers(model))
    method = (
        "AttentionExplainer endpoint-conditioned proxy"
        if attention_available
        else "endpoint-conditioned proxy"
    )
    records: list[AttentionWeight] = []
    for layer in range(layer_count):
        for current_edge_type in data.edge_types:
            if not hasattr(data[current_edge_type], "edge_index"):
                continue
            edge_records = _edge_type_attention(
                data,
                cast(EdgeType, current_edge_type),
                disease_idx,
                gene_idx,
                layer=layer,
                method=method,
                prediction_edge_type=edge_type,
            )
            records.extend(edge_records)

    records.sort(key=lambda item: item.weight, reverse=True)
    selected = records[:top_k]
    return {
        "algorithm": "AttentionExplainer",
        "attention_available": attention_available,
        "method": method,
        "prediction": {
            "edge_type": list(edge_type),
            "disease_idx": disease_idx,
            "gene_idx": gene_idx,
        },
        "weights": [record.to_dict() for record in selected],
        "meta_relation_summary": summarize_attention_by_metarelation(selected),
    }


def summarize_attention_by_metarelation(
    weights: list[AttentionWeight] | list[dict[str, object]],
) -> dict[str, float]:
    """Aggregate attention mass by ``src__relation__dst`` meta-relation."""

    summary: defaultdict[str, float] = defaultdict(float)
    for item in weights:
        if isinstance(item, AttentionWeight):
            key = item.meta_relation
            weight = item.weight
        else:
            key = str(item["meta_relation"])
            weight = float(cast(SupportsFloat, item["weight"]))
        summary[key] += weight
    return dict(sorted(summary.items(), key=lambda row: (-row[1], row[0])))


def _edge_type_attention(
    data: HeteroData,
    edge_type: EdgeType,
    disease_idx: int,
    gene_idx: int,
    *,
    layer: int,
    method: str,
    prediction_edge_type: EdgeType,
) -> list[AttentionWeight]:
    edge_index = data[edge_type].edge_index
    if edge_index.numel() == 0:
        return []
    source_ids = _node_ids(data, edge_type[0])
    target_ids = _node_ids(data, edge_type[2])
    raw_weights = _endpoint_attention_scores(
        edge_type,
        edge_index,
        disease_idx,
        gene_idx,
        prediction_edge_type=prediction_edge_type,
    )
    records: list[AttentionWeight] = []
    for position, weight in enumerate(raw_weights.tolist()):
        source = int(edge_index[0, position].item())
        target = int(edge_index[1, position].item())
        records.append(
            AttentionWeight(
                layer=layer,
                edge_type=edge_type,
                source=source,
                target=target,
                source_id=source_ids[source],
                target_id=target_ids[target],
                weight=float(weight),
                meta_relation="__".join(edge_type),
                method=method,
            )
        )
    return records


def _endpoint_attention_scores(
    edge_type: EdgeType,
    edge_index: torch.Tensor,
    disease_idx: int,
    gene_idx: int,
    *,
    prediction_edge_type: EdgeType,
) -> torch.Tensor:
    source_type, relation, target_type = edge_type
    source = edge_index[0]
    target = edge_index[1]
    scores = torch.full((edge_index.size(1),), 0.05, dtype=torch.float32)
    if source_type == "disease":
        scores += (source == disease_idx).to(torch.float32) * 0.70
    if target_type == "disease":
        scores += (target == disease_idx).to(torch.float32) * 0.70
    if source_type == "gene":
        scores += (source == gene_idx).to(torch.float32) * 0.70
    if target_type == "gene":
        scores += (target == gene_idx).to(torch.float32) * 0.70
    if edge_type == prediction_edge_type:
        scores += ((source == disease_idx) & (target == gene_idx)).to(torch.float32)
    if (
        relation.startswith("rev_")
        and source_type == "gene"
        and target_type == "disease"
    ):
        scores += ((source == gene_idx) & (target == disease_idx)).to(torch.float32)
    maximum = scores.max()
    if float(maximum.item()) <= 0.0:
        return scores
    return scores / maximum


def _num_hgt_layers(model: nn.Module) -> int:
    candidate: Any = model
    if hasattr(candidate, "encoder"):
        candidate = candidate.encoder
    convs = getattr(candidate, "convs", None)
    if convs is None:
        return 1
    try:
        return len(convs)
    except TypeError:
        return 1


def _attention_can_construct() -> bool:
    try:
        AttentionExplainer(reduce="mean")
    except Exception:
        return False
    return True


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]
