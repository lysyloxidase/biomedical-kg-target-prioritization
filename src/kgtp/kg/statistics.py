"""Graph statistics for the Phase 1 heterogeneous KG."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class GraphStatistics:
    """Computed statistics for canonical node and edge tables."""

    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    density: float
    mean_degree: float
    positive_disease_gene_links: int

    @classmethod
    def from_tables(
        cls,
        nodes: pd.DataFrame,
        edge_tables: Mapping[str, pd.DataFrame],
    ) -> GraphStatistics:
        """Compute graph-level statistics from canonical tables."""

        node_counts = (
            nodes.groupby("node_type")["node_id"].nunique().sort_index().to_dict()
        )
        edge_counts = {name: len(edges) for name, edges in sorted(edge_tables.items())}
        edge_total = sum(edge_counts.values())
        node_total = int(nodes["node_id"].nunique())
        density = (
            edge_total / (node_total * (node_total - 1)) if node_total > 1 else 0.0
        )
        mean_degree = (2 * edge_total / node_total) if node_total else 0.0
        positive_links = count_positive_disease_gene_links(edge_tables)
        return cls(
            node_counts={str(key): int(value) for key, value in node_counts.items()},
            edge_counts={str(key): int(value) for key, value in edge_counts.items()},
            density=float(density),
            mean_degree=float(mean_degree),
            positive_disease_gene_links=int(positive_links),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a long-form statistics table."""

        rows: list[dict[str, object]] = []
        rows.extend(
            {"metric": "node_count", "name": name, "value": count}
            for name, count in self.node_counts.items()
        )
        rows.extend(
            {"metric": "edge_count", "name": name, "value": count}
            for name, count in self.edge_counts.items()
        )
        rows.extend(
            [
                {"metric": "density", "name": "global", "value": self.density},
                {"metric": "mean_degree", "name": "global", "value": self.mean_degree},
                {
                    "metric": "positive_disease_gene_links",
                    "name": "Open Targets",
                    "value": self.positive_disease_gene_links,
                },
            ]
        )
        return pd.DataFrame(rows)


def count_positive_disease_gene_links(edge_tables: Mapping[str, pd.DataFrame]) -> int:
    """Count primary positive disease-gene links."""

    disease_gene = edge_tables.get("disease_gene")
    if disease_gene is None or disease_gene.empty:
        return 0
    mask = (
        (disease_gene["source_type"] == "Disease")
        & (disease_gene["edge_type"] == "associated_with")
        & (disease_gene["target_type"] == "Gene")
    )
    return int(mask.sum())


def degree_distribution(edge_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return total degree per node across all edge tables."""

    endpoints: list[pd.Series] = []
    for edges in edge_tables.values():
        endpoints.append(edges["source_id"])
        endpoints.append(edges["target_id"])
    if not endpoints:
        return pd.DataFrame(columns=["node_id", "degree"])
    values = pd.concat(endpoints, ignore_index=True)
    counts = values.value_counts().rename_axis("node_id").reset_index(name="degree")
    sorted_counts = counts.sort_values(
        ["degree", "node_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return cast(pd.DataFrame, sorted_counts)


def known_gene_presence(
    nodes: pd.DataFrame,
    known_genes: list[str] | set[str] | tuple[str, ...],
) -> dict[str, bool]:
    """Check whether known OA genes are represented by symbol or node ID."""

    requested = {gene.upper() for gene in known_genes}
    identifiers = set(nodes["node_id"].astype(str).str.upper())
    if "symbol" in nodes.columns:
        identifiers.update(nodes["symbol"].dropna().astype(str).str.upper())
    return {gene: gene in identifiers for gene in sorted(requested)}


def graph_statistics_table(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Convenience wrapper returning a long-form statistics table."""

    return GraphStatistics.from_tables(nodes, edge_tables).to_frame()
