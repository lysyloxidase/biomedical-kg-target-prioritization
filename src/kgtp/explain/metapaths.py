"""Meta-path and path-based explanations for disease-gene predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch_geometric.data import HeteroData

from kgtp.explain.explainer import DISEASE_GENE_EDGE

EdgeType = tuple[str, str, str]
GENE_PATHWAY_EDGE: EdgeType = ("gene", "participates_in", "pathway")
GENE_PPI_EDGE: EdgeType = ("gene", "interacts", "gene")


@dataclass(frozen=True)
class PathNode:
    """A typed node in a meta-path explanation."""

    node_type: str
    index: int
    node_id: str
    label: str


@dataclass(frozen=True)
class MetaPathExplanation:
    """Ranked multi-hop path connecting disease and predicted gene."""

    schema: str
    nodes: tuple[PathNode, ...]
    edge_types: tuple[EdgeType, ...]
    score: float
    evidence: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["edge_types"] = [list(edge_type) for edge_type in self.edge_types]
        return payload


def rank_metapaths(
    data: HeteroData,
    disease_idx: int,
    gene_idx: int,
    *,
    max_paths: int = 10,
) -> list[MetaPathExplanation]:
    """Surface disease-gene-pathway-gene and disease-gene-PPI-gene paths."""

    paths = [
        *_shared_pathway_paths(data, disease_idx, gene_idx),
        *_ppi_paths(data, disease_idx, gene_idx),
    ]
    paths.sort(key=lambda item: (item.score, item.schema, item.evidence), reverse=True)
    return paths[:max_paths]


def metapaths_to_records(paths: list[MetaPathExplanation]) -> list[dict[str, object]]:
    """Serialize meta-path explanations for reports and tests."""

    return [path.to_dict() for path in paths]


def _shared_pathway_paths(
    data: HeteroData,
    disease_idx: int,
    gene_idx: int,
) -> list[MetaPathExplanation]:
    if (
        DISEASE_GENE_EDGE not in data.edge_types
        or GENE_PATHWAY_EDGE not in data.edge_types
    ):
        return []

    associated_genes = _targets_for_source(
        data[DISEASE_GENE_EDGE].edge_index, disease_idx
    )
    gene_to_pathways = _targets_by_source(data[GENE_PATHWAY_EDGE].edge_index)
    predicted_pathways = gene_to_pathways.get(gene_idx, set())
    if not associated_genes or not predicted_pathways:
        return []

    disease_node = _path_node(data, "disease", disease_idx)
    predicted_gene_node = _path_node(data, "gene", gene_idx)
    pathway_sizes = _source_counts_by_target(data[GENE_PATHWAY_EDGE].edge_index)
    out: list[MetaPathExplanation] = []
    for intermediate_gene in sorted(associated_genes):
        shared = predicted_pathways & gene_to_pathways.get(intermediate_gene, set())
        for pathway_idx in sorted(shared):
            compactness = 1.0 / max(1.0, float(pathway_sizes[pathway_idx]))
            score = 1.0 + compactness
            out.append(
                MetaPathExplanation(
                    schema="disease->gene->pathway->gene",
                    nodes=(
                        disease_node,
                        _path_node(data, "gene", intermediate_gene),
                        _path_node(data, "pathway", pathway_idx),
                        predicted_gene_node,
                    ),
                    edge_types=(
                        DISEASE_GENE_EDGE,
                        GENE_PATHWAY_EDGE,
                        GENE_PATHWAY_EDGE,
                    ),
                    score=score,
                    evidence="shared_pathway",
                )
            )
    return out


def _ppi_paths(
    data: HeteroData,
    disease_idx: int,
    gene_idx: int,
) -> list[MetaPathExplanation]:
    if DISEASE_GENE_EDGE not in data.edge_types or GENE_PPI_EDGE not in data.edge_types:
        return []

    associated_genes = _targets_for_source(
        data[DISEASE_GENE_EDGE].edge_index, disease_idx
    )
    ppi_neighbors = _undirected_neighbors(data[GENE_PPI_EDGE].edge_index, gene_idx)
    disease_ppi_neighbors = sorted(associated_genes & ppi_neighbors)
    disease_node = _path_node(data, "disease", disease_idx)
    predicted_gene_node = _path_node(data, "gene", gene_idx)
    degrees = _undirected_degrees(data[GENE_PPI_EDGE].edge_index)
    out: list[MetaPathExplanation] = []
    for intermediate_gene in disease_ppi_neighbors:
        compactness = 1.0 / max(1.0, float(degrees[intermediate_gene]))
        out.append(
            MetaPathExplanation(
                schema="disease->gene->PPI->gene",
                nodes=(
                    disease_node,
                    _path_node(data, "gene", intermediate_gene),
                    predicted_gene_node,
                ),
                edge_types=(DISEASE_GENE_EDGE, GENE_PPI_EDGE),
                score=0.9 + compactness,
                evidence="ppi_neighbor_of_known_target",
            )
        )
    return out


def _path_node(data: HeteroData, node_type: str, index: int) -> PathNode:
    node_ids = _node_ids(data, node_type)
    labels = _node_labels(data, node_type, node_ids)
    return PathNode(
        node_type=node_type,
        index=index,
        node_id=node_ids[index],
        label=labels[index],
    )


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]


def _node_labels(data: HeteroData, node_type: str, node_ids: list[str]) -> list[str]:
    for attr in ("symbol", "label", "name", "node_label"):
        if hasattr(data[node_type], attr):
            values = getattr(data[node_type], attr)
            return [str(value) for value in values]
    return node_ids


def _targets_for_source(edge_index: torch.Tensor, source_idx: int) -> set[int]:
    return {
        int(target)
        for source, target in edge_index.t().tolist()
        if int(source) == source_idx
    }


def _targets_by_source(edge_index: torch.Tensor) -> dict[int, set[int]]:
    out: defaultdict[int, set[int]] = defaultdict(set)
    for source, target in edge_index.t().tolist():
        out[int(source)].add(int(target))
    return dict(out)


def _source_counts_by_target(edge_index: torch.Tensor) -> dict[int, int]:
    out: defaultdict[int, int] = defaultdict(int)
    for _, target in edge_index.t().tolist():
        out[int(target)] += 1
    return dict(out)


def _undirected_neighbors(edge_index: torch.Tensor, node_idx: int) -> set[int]:
    neighbors: set[int] = set()
    for source, target in edge_index.t().tolist():
        source_int = int(source)
        target_int = int(target)
        if source_int == node_idx:
            neighbors.add(target_int)
        if target_int == node_idx:
            neighbors.add(source_int)
    return neighbors


def _undirected_degrees(edge_index: torch.Tensor) -> dict[int, int]:
    out: defaultdict[int, int] = defaultdict(int)
    for source, target in edge_index.t().tolist():
        out[int(source)] += 1
        out[int(target)] += 1
    return cast(dict[int, int], dict(out))
