"""Model attributions for heterogeneous link prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, SupportsFloat, cast

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import (
    AttentionExplainer,
    CaptumExplainer,
    Explainer,
    HeteroExplanation,
)

EdgeType = tuple[str, str, str]
DISEASE_GENE_EDGE: EdgeType = ("disease", "associated_with", "gene")


class TargetExplainer:
    """Explain one predicted disease-to-gene link on a ``HeteroData`` graph."""

    def __init__(
        self,
        model: nn.Module,
        data: HeteroData | None = None,
        *,
        edge_type: EdgeType = DISEASE_GENE_EDGE,
        explanation_type: str = "model",
        integration_steps: int = 16,
        use_pyg_captum: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.edge_type = edge_type
        self.explanation_type = explanation_type
        self.integration_steps = max(1, integration_steps)
        self.use_pyg_captum = use_pyg_captum
        self.algorithm_name = "CaptumExplainer"
        self.attribution_method = "IntegratedGradients"
        self.attention_algorithm_name = "AttentionExplainer"
        self.last_pyg_error: str | None = None
        self.captum_available = self._captum_can_construct()
        self.attention_available = self._attention_can_construct()

    def explain_link(
        self,
        disease_idx: int,
        gene_idx: int,
        data: HeteroData | None = None,
    ) -> HeteroExplanation:
        """Return feature/node/edge attributions for one disease-gene link."""

        graph = self._require_data(data)
        self._require_trained_model()
        edge_label_index = torch.tensor([[disease_idx], [gene_idx]], dtype=torch.long)

        if self.use_pyg_captum and self.captum_available:
            explanation = self._try_pyg_captum(graph, edge_label_index)
            if explanation is not None:
                return self._annotate_explanation(
                    explanation,
                    graph,
                    disease_idx,
                    gene_idx,
                    method="CaptumExplainer(IntegratedGradients)",
                )

        attributions, score = self._integrated_gradients(graph, edge_label_index)
        edge_masks = self._edge_occlusion_masks(
            graph,
            edge_label_index,
            baseline_score=score,
        )
        explanation = self._build_hetero_explanation(
            graph,
            attributions,
            edge_masks,
            disease_idx=disease_idx,
            gene_idx=gene_idx,
            score=score,
            method="IntegratedGradients features + edge occlusion",
        )
        return explanation

    def score_link(
        self,
        disease_idx: int,
        gene_idx: int,
        data: HeteroData | None = None,
    ) -> float:
        """Return the trained model's raw score for one candidate link."""
        graph = self._require_data(data)
        self._require_trained_model()
        index = torch.tensor([[disease_idx], [gene_idx]], dtype=torch.long)
        return self._safe_no_grad_score(graph, index)

    def score_without_edges(
        self,
        disease_idx: int,
        gene_idx: int,
        removals: Mapping[EdgeType, list[int]],
        data: HeteroData | None = None,
    ) -> float:
        """Score a link after removing selected message-edge positions."""
        graph = self._require_data(data)
        self._require_trained_model()
        edge_index_dict = _edge_index_dict(graph)
        for edge_type, positions in removals.items():
            edge_index = edge_index_dict.get(edge_type)
            if edge_index is None or not positions:
                continue
            keep = torch.ones(edge_index.size(1), dtype=torch.bool)
            keep[torch.tensor(positions, dtype=torch.long)] = False
            edge_index_dict[edge_type] = edge_index[:, keep]
        index = torch.tensor([[disease_idx], [gene_idx]], dtype=torch.long)
        with torch.no_grad():
            return float(self._link_score(graph.x_dict, edge_index_dict, index).item())

    def explanatory_subgraph(
        self,
        explanation: HeteroExplanation,
        *,
        top_k_nodes: int = 8,
        top_k_edges: int = 12,
    ) -> dict[str, object]:
        """Extract compact, serializable node/edge rationale records."""

        nodes: list[dict[str, object]] = []
        for node_type in explanation.node_types:
            store = explanation[node_type]
            if not hasattr(store, "node_mask"):
                continue
            node_mask = cast(torch.Tensor, store.node_mask).detach().view(-1)
            node_ids = _node_ids_from_store(store, node_mask.numel())
            top_values = _topk(node_mask, top_k_nodes)
            for index, importance in top_values:
                nodes.append(
                    {
                        "node_type": node_type,
                        "index": index,
                        "node_id": node_ids[index],
                        "importance": importance,
                    }
                )

        edges: list[dict[str, object]] = []
        for edge_type in explanation.edge_types:
            store = explanation[edge_type]
            if not hasattr(store, "edge_index") or not hasattr(store, "edge_mask"):
                continue
            edge_index = cast(torch.Tensor, store.edge_index)
            edge_mask = cast(torch.Tensor, store.edge_mask).detach().view(-1)
            for edge_pos, importance in _topk(edge_mask, top_k_edges):
                edges.append(
                    {
                        "edge_type": list(edge_type),
                        "source": int(edge_index[0, edge_pos].item()),
                        "target": int(edge_index[1, edge_pos].item()),
                        "importance": importance,
                    }
                )

        feature_importance: dict[str, list[dict[str, float | int]]] = {}
        for node_type in explanation.node_types:
            store = explanation[node_type]
            if not hasattr(store, "feature_mask"):
                continue
            feature_mask = cast(torch.Tensor, store.feature_mask).detach().view(-1)
            feature_importance[node_type] = [
                {"feature_index": index, "importance": importance}
                for index, importance in _topk(
                    feature_mask, min(8, feature_mask.numel())
                )
            ]

        nodes.sort(
            key=lambda row: float(cast(SupportsFloat, row["importance"])), reverse=True
        )
        edges.sort(
            key=lambda row: float(cast(SupportsFloat, row["importance"])), reverse=True
        )
        return {
            "method": getattr(explanation, "method", "unknown"),
            "model_attribution": True,
            "causal_explanation": False,
            "score": float(getattr(explanation, "score", math.nan)),
            "prediction": {
                "edge_type": list(
                    getattr(explanation, "prediction_edge_type", self.edge_type)
                ),
                "disease_idx": int(getattr(explanation, "disease_idx", -1)),
                "gene_idx": int(getattr(explanation, "gene_idx", -1)),
            },
            "nodes": nodes[:top_k_nodes],
            "edges": edges[:top_k_edges],
            "feature_importance": feature_importance,
        }

    def _try_pyg_captum(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
    ) -> HeteroExplanation | None:
        try:
            wrapper = _LinkScoreWrapper(self.model, self.edge_type)
            explainer = Explainer(
                model=wrapper,
                algorithm=CaptumExplainer(self.attribution_method),
                explanation_type=self.explanation_type,
                model_config={
                    "mode": "binary_classification",
                    "task_level": "edge",
                    "return_type": "raw",
                },
                node_mask_type="attributes",
                edge_mask_type="object",
            )
            explanation = explainer(
                data.x_dict,
                _edge_index_dict(data),
                edge_label_index=edge_label_index,
                index=0,
            )
        except Exception as exc:  # pragma: no cover - PyG/Captum version surface.
            self.last_pyg_error = f"{type(exc).__name__}: {exc}"
            return None
        if isinstance(explanation, HeteroExplanation):
            return explanation
        return None

    def _integrated_gradients(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], float]:
        base_x = {
            node_type: data[node_type].x.detach().to(torch.float32)
            for node_type in data.node_types
        }
        baselines = {
            node_type: torch.zeros_like(features)
            for node_type, features in base_x.items()
        }
        accumulated = {
            node_type: torch.zeros_like(features)
            for node_type, features in base_x.items()
        }
        node_order = list(base_x)
        score_value = math.nan

        self.model.eval()
        try:
            for step in range(1, self.integration_steps + 1):
                alpha = step / self.integration_steps
                scaled = {
                    node_type: (
                        baselines[node_type]
                        + alpha * (base_x[node_type] - baselines[node_type])
                    )
                    .detach()
                    .requires_grad_(True)
                    for node_type in node_order
                }
                self.model.zero_grad(set_to_none=True)
                score = self._link_score(
                    scaled, _edge_index_dict(data), edge_label_index
                )
                score_value = float(score.detach().item())
                grads = torch.autograd.grad(
                    score,
                    tuple(scaled[node_type] for node_type in node_order),
                    allow_unused=True,
                )
                for node_type, grad in zip(node_order, grads, strict=True):
                    if grad is not None:
                        accumulated[node_type] = accumulated[node_type] + grad.detach()
            attributions = {
                node_type: (base_x[node_type] - baselines[node_type])
                * (accumulated[node_type] / self.integration_steps)
                for node_type in node_order
            }
        except Exception as exc:
            self.last_pyg_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                "Integrated Gradients failed; no heuristic explanation was emitted"
            ) from exc
        return attributions, score_value

    def _edge_occlusion_masks(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        *,
        baseline_score: float,
    ) -> dict[EdgeType, torch.Tensor]:
        masks: dict[EdgeType, torch.Tensor] = {}
        disease_idx = int(edge_label_index[0, 0].item())
        gene_idx = int(edge_label_index[1, 0].item())
        edge_index_dict = _edge_index_dict(data)
        for edge_type, edge_index in edge_index_dict.items():
            mask = torch.zeros(edge_index.size(1), dtype=torch.float32)
            for position in _local_edge_positions(
                edge_type,
                edge_index,
                disease_idx,
                gene_idx,
            ):
                perturbed = dict(edge_index_dict)
                keep = torch.ones(edge_index.size(1), dtype=torch.bool)
                keep[position] = False
                perturbed[edge_type] = edge_index[:, keep]
                with torch.no_grad():
                    score = self._link_score(
                        data.x_dict,
                        perturbed,
                        edge_label_index,
                    )
                mask[position] = abs(baseline_score - float(score.item()))
            masks[edge_type] = _normalize(mask)
        return masks

    def _link_score(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.model, "encode") and hasattr(self.model, "decode"):
            z_dict = cast(Any, self.model).encode(x_dict, edge_index_dict)
            logits = cast(Any, self.model).decode(
                z_dict, self.edge_type, edge_label_index
            )
            return cast(torch.Tensor, logits).view(-1)[0]

        output = self.model(x_dict, edge_index_dict, {self.edge_type: edge_label_index})
        if isinstance(output, Mapping):
            return cast(torch.Tensor, output[self.edge_type]).view(-1)[0]
        return cast(torch.Tensor, output).view(-1)[0]

    def _safe_no_grad_score(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            try:
                score = self._link_score(
                    data.x_dict, _edge_index_dict(data), edge_label_index
                )
            except Exception:
                return math.nan
        return float(score.item())

    def _build_hetero_explanation(
        self,
        data: HeteroData,
        attributions: Mapping[str, torch.Tensor],
        edge_masks: Mapping[EdgeType, torch.Tensor],
        *,
        disease_idx: int,
        gene_idx: int,
        score: float,
        method: str,
    ) -> HeteroExplanation:
        explanation = HeteroExplanation()
        for node_type in data.node_types:
            attribution = attributions[node_type].detach().abs()
            store = explanation[node_type]
            store.x = data[node_type].x.detach()
            store.node_id = _node_ids(data, node_type)
            store.raw_feature_attribution = attributions[node_type].detach()
            store.node_mask = _normalize(attribution.sum(dim=1, keepdim=True))
            store.feature_mask = _normalize(attribution.mean(dim=0))

        for edge_type in data.edge_types:
            if not hasattr(data[edge_type], "edge_index"):
                continue
            edge_index = data[edge_type].edge_index.detach()
            store = explanation[edge_type]
            store.edge_index = edge_index
            store.edge_mask = edge_masks[cast(EdgeType, edge_type)]

        return self._annotate_explanation(
            explanation,
            data,
            disease_idx,
            gene_idx,
            method=method,
            score=score,
        )

    def _annotate_explanation(
        self,
        explanation: HeteroExplanation,
        data: HeteroData,
        disease_idx: int,
        gene_idx: int,
        *,
        method: str,
        score: float | None = None,
    ) -> HeteroExplanation:
        explanation.method = method
        explanation.algorithm = self.algorithm_name
        explanation.attribution_method = self.attribution_method
        explanation.edge_attribution_method = "single-edge occlusion"
        explanation.model_attribution = True
        explanation.causal_explanation = False
        explanation.prediction_edge_type = self.edge_type
        explanation.disease_idx = disease_idx
        explanation.gene_idx = gene_idx
        explanation.disease_id = _node_ids(data, self.edge_type[0])[disease_idx]
        explanation.gene_id = _node_ids(data, self.edge_type[2])[gene_idx]
        if score is not None:
            explanation.score = score
        elif not hasattr(explanation, "score"):
            explanation.score = self._safe_no_grad_score(
                data,
                torch.tensor([[disease_idx], [gene_idx]], dtype=torch.long),
            )
        return explanation

    def _require_data(self, data: HeteroData | None) -> HeteroData:
        graph = data if data is not None else self.data
        if graph is None:
            msg = "TargetExplainer requires HeteroData at construction or call time"
            raise ValueError(msg)
        return graph

    def _require_trained_model(self) -> None:
        if getattr(self.model, "is_trained", False) is not True:
            raise RuntimeError("Explanations require a validated trained model")

    def _captum_can_construct(self) -> bool:
        try:
            CaptumExplainer(self.attribution_method)
        except Exception:
            return False
        return True

    def _attention_can_construct(self) -> bool:
        try:
            AttentionExplainer(reduce="mean")
        except Exception:
            return False
        return True


class _LinkScoreWrapper(nn.Module):
    """Adapt the project multitask model to PyG's edge-level explainer API."""

    def __init__(self, model: nn.Module, edge_type: EdgeType) -> None:
        super().__init__()
        self.model = model
        self.edge_type = edge_type

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.model, "encode") and hasattr(self.model, "decode"):
            z_dict = cast(Any, self.model).encode(x_dict, edge_index_dict)
            return cast(
                torch.Tensor,
                cast(Any, self.model).decode(z_dict, self.edge_type, edge_label_index),
            )
        output = self.model(x_dict, edge_index_dict, {self.edge_type: edge_label_index})
        if isinstance(output, Mapping):
            return cast(torch.Tensor, output[self.edge_type])
        return cast(torch.Tensor, output)


def _edge_index_dict(data: HeteroData) -> dict[EdgeType, torch.Tensor]:
    return {
        cast(EdgeType, edge_type): data[edge_type].edge_index
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]


def _node_ids_from_store(store: Any, size: int) -> list[str]:
    if hasattr(store, "node_id"):
        return [str(value) for value in store.node_id]
    return [str(index) for index in range(size)]


def _local_edge_positions(
    edge_type: EdgeType,
    edge_index: torch.Tensor,
    disease_idx: int,
    gene_idx: int,
) -> list[int]:
    if edge_index.numel() == 0:
        return []
    src_type, _, dst_type = edge_type
    source = edge_index[0]
    target = edge_index[1]
    selected = torch.zeros(edge_index.size(1), dtype=torch.bool)
    if src_type == "disease":
        selected |= source == disease_idx
    if dst_type == "disease":
        selected |= target == disease_idx
    if src_type == "gene":
        selected |= source == gene_idx
    if dst_type == "gene":
        selected |= target == gene_idx
    return selected.nonzero(as_tuple=False).view(-1).tolist()


def _normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.to(torch.float32)
    values = values.to(torch.float32)
    maximum = values.max()
    if float(maximum.item()) <= 0.0:
        return torch.zeros_like(values)
    return values / maximum


def _topk(values: torch.Tensor, k: int) -> list[tuple[int, float]]:
    if values.numel() == 0 or k <= 0:
        return []
    count = min(k, values.numel())
    top = torch.topk(values.detach().to(torch.float32), count)
    return [
        (int(index.item()), float(value.item()))
        for index, value in zip(top.indices, top.values, strict=True)
    ]
