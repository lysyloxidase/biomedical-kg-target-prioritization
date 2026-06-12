"""Feature transformers fitted exclusively on an allowed message graph."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, Self, cast

import pandas as pd
import torch


class FeatureTransformer(Protocol):
    """Contract for stateful graph feature construction."""

    def fit(
        self,
        nodes: pd.DataFrame,
        edge_tables: Mapping[str, pd.DataFrame],
        *,
        graph_hash: str,
    ) -> Self:
        """Fit vocabulary and topology-derived state on one allowed graph."""
        ...

    def transform(self, nodes: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Transform graph entities without inspecting graph edges."""
        ...


@dataclass
class TrainGraphFeatureTransformer:
    """Build structural and ontology features from the train message graph."""

    max_go_terms: int = 512
    fitted_graph_hash: str | None = None
    go_vocabulary: tuple[str, ...] = ()
    features: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def fit(
        self,
        nodes: pd.DataFrame,
        edge_tables: Mapping[str, pd.DataFrame],
        *,
        graph_hash: str,
    ) -> Self:
        """Fit all graph-dependent state using only the supplied graph."""

        gene_ids = _node_ids(nodes, "Gene")
        pathway_ids = _node_ids(nodes, "Pathway")
        go_ids = _node_ids(nodes, "GOTerm")
        degree, pagerank = _degree_and_pagerank(edge_tables)
        gene_go = edge_tables.get("gene_go")
        self.go_vocabulary = tuple(
            sorted(
                set() if gene_go is None else gene_go["target_id"].astype(str).unique()
            )[: self.max_go_terms]
        )
        annotations = _gene_go_annotations(gene_go)
        term_index = {term: index for index, term in enumerate(self.go_vocabulary)}

        gene_features: dict[str, list[float]] = {}
        for gene_id in gene_ids:
            multihot = [0.0] * len(self.go_vocabulary)
            for go_id in annotations.get(gene_id, set()):
                if go_id in term_index:
                    multihot[term_index[go_id]] = 1.0
            gene_features[gene_id] = [
                float(degree.get(gene_id, 0)),
                float(pagerank.get(gene_id, 0.0)),
                *multihot,
            ]

        participant_counts = _pathway_participant_counts(
            edge_tables.get("gene_pathway")
        )
        pathway_depths = _pathway_depths(edge_tables.get("pathway_pathway"))
        pathway_features = {
            pathway_id: [
                float(participant_counts.get(pathway_id, 0)),
                float(pathway_depths.get(pathway_id, 0)),
            ]
            for pathway_id in pathway_ids
        }

        node_rows = _node_rows(nodes)
        annotation_counts = _annotation_counts(gene_go)
        total_annotations = sum(annotation_counts.values())
        go_features = {
            go_id: [
                *_namespace_one_hot(node_rows.get(("GOTerm", go_id), {})),
                _information_content(
                    annotation_counts.get(go_id, 0), total_annotations
                ),
            ]
            for go_id in go_ids
        }

        self.features = {
            "gene": gene_features,
            "pathway": pathway_features,
            "go_term": go_features,
        }
        self.fitted_graph_hash = graph_hash
        return self

    def transform(self, nodes: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Return fitted features in deterministic node-index order."""

        if self.fitted_graph_hash is None or not self.features:
            msg = "Feature transformer must be fitted before transform"
            raise RuntimeError(msg)
        result: dict[str, torch.Tensor] = {}
        for pyg_type, canonical_type in (
            ("gene", "Gene"),
            ("pathway", "Pathway"),
            ("go_term", "GOTerm"),
        ):
            ids = _node_ids(nodes, canonical_type)
            feature_map = self.features[pyg_type]
            missing = [node_id for node_id in ids if node_id not in feature_map]
            if missing:
                msg = (
                    f"Transformer has no fitted {pyg_type} features for: "
                    + ", ".join(missing[:5])
                )
                raise ValueError(msg)
            result[pyg_type] = torch.tensor(
                [feature_map[node_id] for node_id in ids],
                dtype=torch.float32,
            )
        return result

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted state without Python pickles."""

        if self.fitted_graph_hash is None:
            msg = "Cannot serialize an unfitted feature transformer"
            raise RuntimeError(msg)
        return {
            "schema_version": 1,
            "transformer": type(self).__name__,
            "fit_scope": "train_message_graph_only",
            "fitted_graph_hash": self.fitted_graph_hash,
            "max_go_terms": self.max_go_terms,
            "go_vocabulary": list(self.go_vocabulary),
            "features": self.features,
            "state_hash": self.state_hash(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrainGraphFeatureTransformer:
        """Restore a transformer from its JSON representation."""

        raw_features = payload.get("features")
        if not isinstance(raw_features, dict):
            msg = "Serialized transformer is missing feature state"
            raise ValueError(msg)
        transformer = cls(max_go_terms=int(cast(int, payload.get("max_go_terms", 512))))
        transformer.fitted_graph_hash = str(payload["fitted_graph_hash"])
        raw_vocabulary = payload.get("go_vocabulary", [])
        if not isinstance(raw_vocabulary, list):
            msg = "Serialized transformer has an invalid GO vocabulary"
            raise ValueError(msg)
        transformer.go_vocabulary = tuple(str(value) for value in raw_vocabulary)
        features: dict[str, dict[str, list[float]]] = {}
        for node_type, feature_map in raw_features.items():
            if not isinstance(feature_map, dict):
                msg = f"Invalid feature map for {node_type}"
                raise ValueError(msg)
            features[str(node_type)] = {
                str(node_id): [float(value) for value in values]
                for node_id, values in feature_map.items()
                if isinstance(values, list)
            }
        transformer.features = features
        if payload.get("state_hash") != transformer.state_hash():
            msg = "Serialized feature-transformer state hash does not match"
            raise ValueError(msg)
        return transformer

    def state_hash(self) -> str:
        """Hash the fitted state for artifact compatibility checks."""

        payload = {
            "fitted_graph_hash": self.fitted_graph_hash,
            "max_go_terms": self.max_go_terms,
            "go_vocabulary": list(self.go_vocabulary),
            "features": self.features,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _node_ids(nodes: pd.DataFrame, node_type: str) -> list[str]:
    return sorted(
        nodes.loc[nodes["node_type"] == node_type, "node_id"].astype(str).unique()
    )


def _node_rows(
    nodes: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(row["node_type"]), str(row["node_id"])): {
            str(key): cast(object, value)
            for key, value in row.dropna().to_dict().items()
        }
        for _, row in nodes.iterrows()
    }


def _degree_and_pagerank(
    edge_tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, int], dict[str, float]]:
    degree: defaultdict[str, int] = defaultdict(int)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for edges in edge_tables.values():
        for source_id, target_id in zip(
            edges["source_id"], edges["target_id"], strict=True
        ):
            source = str(source_id)
            target = str(target_id)
            degree[source] += 1
            degree[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)
    return dict(degree), _pagerank(adjacency)


def _pagerank(
    adjacency: Mapping[str, set[str]],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    nodes = sorted(adjacency)
    if not nodes:
        return {}
    scores = dict.fromkeys(nodes, 1.0 / len(nodes))
    for _ in range(iterations):
        updated = {node: (1.0 - damping) / len(nodes) for node in nodes}
        for source in nodes:
            neighbors = adjacency[source]
            if not neighbors:
                continue
            contribution = damping * scores[source] / len(neighbors)
            for target in neighbors:
                updated[target] += contribution
        scores = updated
    return scores


def _gene_go_annotations(
    gene_go: pd.DataFrame | None,
) -> dict[str, set[str]]:
    annotations: defaultdict[str, set[str]] = defaultdict(set)
    if gene_go is not None:
        for gene_id, go_id in zip(
            gene_go["source_id"], gene_go["target_id"], strict=True
        ):
            annotations[str(gene_id)].add(str(go_id))
    return dict(annotations)


def _pathway_participant_counts(
    gene_pathway: pd.DataFrame | None,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    if gene_pathway is not None:
        for pathway_id in gene_pathway["target_id"].astype(str):
            counts[pathway_id] += 1
    return dict(counts)


def _pathway_depths(
    hierarchy: pd.DataFrame | None,
) -> dict[str, int]:
    if hierarchy is None or hierarchy.empty:
        return {}
    children: defaultdict[str, set[str]] = defaultdict(set)
    parents: defaultdict[str, set[str]] = defaultdict(set)
    pathways: set[str] = set()
    for parent_id, child_id in zip(
        hierarchy["source_id"], hierarchy["target_id"], strict=True
    ):
        parent = str(parent_id)
        child = str(child_id)
        children[parent].add(child)
        parents[child].add(parent)
        pathways.update((parent, child))
    queue = deque((pathway, 0) for pathway in sorted(pathways) if not parents[pathway])
    depths: dict[str, int] = {}
    while queue:
        pathway, depth = queue.popleft()
        if pathway in depths and depths[pathway] <= depth:
            continue
        depths[pathway] = depth
        queue.extend((child, depth + 1) for child in sorted(children[pathway]))
    return depths


def _annotation_counts(gene_go: pd.DataFrame | None) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    if gene_go is not None:
        for go_id in gene_go["target_id"].astype(str):
            counts[go_id] += 1
    return dict(counts)


def _namespace_one_hot(row: Mapping[str, object]) -> list[float]:
    text = str(row.get("namespace") or row.get("aspect") or "BP").lower()
    if text in {"p", "bp", "biological_process"}:
        namespace = "BP"
    elif text in {"f", "mf", "molecular_function"}:
        namespace = "MF"
    elif text in {"c", "cc", "cellular_component"}:
        namespace = "CC"
    else:
        namespace = "BP"
    return [1.0 if namespace == value else 0.0 for value in ("BP", "MF", "CC")]


def _information_content(count: int, total: int) -> float:
    if count == 0 or total == 0:
        return 0.0
    return float(-math.log(count / total))
