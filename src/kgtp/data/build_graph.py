"""Assemble OA-centric heterogeneous edge and node tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from kgtp.data.common import (
    PathLike,
    normalize_ensembl_gene_id,
    stable_drop_duplicates,
    write_table,
)

EdgeSchema = tuple[str, str, str]

CANONICAL_EDGE_SCHEMAS: dict[str, EdgeSchema] = {
    "disease_gene": ("Disease", "associated_with", "Gene"),
    "gene_gene": ("Gene", "interacts", "Gene"),
    "gene_pathway": ("Gene", "participates_in", "Pathway"),
    "drug_gene": ("Drug", "targets", "Gene"),
    "gene_go": ("Gene", "annotated_with", "GOTerm"),
    "pathway_pathway": ("Pathway", "parent_of", "Pathway"),
}

REQUIRED_EDGE_TABLES = (
    "disease_gene",
    "gene_gene",
    "gene_pathway",
    "drug_gene",
    "gene_go",
)


def make_edge_table(
    rows: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    source_type: str,
    edge_type: str,
    target_type: str,
    source_name: str,
    property_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a canonical edge table from source/target columns."""

    property_columns = property_columns or []
    out = pd.DataFrame(
        {
            "source_id": rows[source_column].astype(str),
            "target_id": rows[target_column].astype(str),
            "source_type": source_type,
            "edge_type": edge_type,
            "target_type": target_type,
            "source": source_name,
        }
    )
    for column in property_columns:
        if column in rows.columns:
            out[column] = rows[column]
    return stable_drop_duplicates(
        out, ["source_id", "target_id", "source_type", "edge_type", "target_type"]
    )


def assemble_canonical_edges(
    *,
    disease_gene: pd.DataFrame,
    gene_gene: pd.DataFrame,
    gene_pathway: pd.DataFrame,
    drug_gene: pd.DataFrame,
    gene_go: pd.DataFrame,
    pathway_pathway: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble all normalized Phase 1 edge tables."""

    edge_tables = {
        "disease_gene": make_edge_table(
            disease_gene,
            source_column="disease_id",
            target_column="gene_id",
            source_type="Disease",
            edge_type="associated_with",
            target_type="Gene",
            source_name="Open Targets",
            property_columns=["score"],
        ),
        "gene_gene": make_edge_table(
            gene_gene,
            source_column="gene_a",
            target_column="gene_b",
            source_type="Gene",
            edge_type="interacts",
            target_type="Gene",
            source_name="STRING",
            property_columns=["score"],
        ),
        "gene_pathway": make_edge_table(
            gene_pathway,
            source_column="gene_id",
            target_column="pathway_id",
            source_type="Gene",
            edge_type="participates_in",
            target_type="Pathway",
            source_name="Reactome",
            property_columns=["pathway_name"],
        ),
        "drug_gene": make_edge_table(
            drug_gene,
            source_column="drug_id",
            target_column="gene_id",
            source_type="Drug",
            edge_type="targets",
            target_type="Gene",
            source_name="ChEMBL",
            property_columns=["target_chembl_id", "action_type", "mechanism_of_action"],
        ),
        "gene_go": make_edge_table(
            gene_go,
            source_column="gene_id",
            target_column="go_id",
            source_type="Gene",
            edge_type="annotated_with",
            target_type="GOTerm",
            source_name="GOA",
            property_columns=["evidence_code"],
        ),
    }
    if pathway_pathway is not None:
        edge_tables["pathway_pathway"] = make_edge_table(
            pathway_pathway,
            source_column="parent_pathway_id",
            target_column="child_pathway_id",
            source_type="Pathway",
            edge_type="parent_of",
            target_type="Pathway",
            source_name="Reactome",
        )
    return edge_tables


def build_node_table(
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    attributes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a deduplicated node table from canonical edge tables."""

    node_frames: list[pd.DataFrame] = []
    for edges in edge_tables.values():
        node_frames.append(
            edges[["source_id", "source_type"]].rename(
                columns={"source_id": "node_id", "source_type": "node_type"}
            )
        )
        node_frames.append(
            edges[["target_id", "target_type"]].rename(
                columns={"target_id": "node_id", "target_type": "node_type"}
            )
        )
    nodes = pd.concat(node_frames, ignore_index=True).drop_duplicates()
    nodes = nodes.sort_values(["node_type", "node_id"]).reset_index(drop=True)
    nodes["label"] = nodes["node_id"]

    if attributes is None or attributes.empty:
        return nodes
    return nodes.merge(attributes, on=["node_id", "node_type"], how="left")


def validate_canonical_edge_types(
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    require_all: bool = True,
) -> list[str]:
    """Return validation errors for canonical edge-table endpoints."""

    errors: list[str] = []
    if require_all:
        missing = [key for key in REQUIRED_EDGE_TABLES if key not in edge_tables]
        errors.extend(f"missing required edge table: {key}" for key in missing)

    for name, edges in edge_tables.items():
        table_errors: list[str] = []
        schema = CANONICAL_EDGE_SCHEMAS.get(name)
        if schema is None:
            errors.append(f"unknown edge table: {name}")
            continue
        expected_source, expected_edge, expected_target = schema
        for column in ("source_type", "edge_type", "target_type"):
            if column not in edges.columns:
                table_errors.append(f"{name} missing {column}")
                continue
        if table_errors:
            errors.extend(table_errors)
            continue
        triples = set(
            zip(
                edges["source_type"],
                edges["edge_type"],
                edges["target_type"],
                strict=False,
            )
        )
        expected = (expected_source, expected_edge, expected_target)
        if triples != {expected}:
            errors.append(f"{name} endpoints {sorted(triples)} != {expected}")
    return errors


def assert_canonical_edge_types(edge_tables: Mapping[str, pd.DataFrame]) -> None:
    """Raise if canonical edge tables are missing or malformed."""

    errors = validate_canonical_edge_types(edge_tables)
    if errors:
        raise ValueError("; ".join(errors))


def check_node_range(nodes: pd.DataFrame, *, min_nodes: int, max_nodes: int) -> bool:
    """Check the configured production graph-size gate."""

    count = len(nodes)
    return min_nodes <= count <= max_nodes


def write_graph_tables(
    edge_tables: Mapping[str, pd.DataFrame],
    nodes: pd.DataFrame,
    output_dir: PathLike,
) -> None:
    """Write node and edge tables under `output_dir`."""

    output = Path(output_dir)
    write_table(nodes, output / "nodes.parquet")
    edges_dir = output / "edges"
    for name, edges in edge_tables.items():
        write_table(edges, edges_dir / f"{name}.parquet")


def normalize_gene_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    """Normalize optional gene attributes for node-table merging."""

    if attributes.empty:
        return pd.DataFrame(columns=["node_id", "node_type"])
    gene_col = "gene_id" if "gene_id" in attributes.columns else "node_id"
    normalized = attributes.copy()
    normalized["node_id"] = normalized[gene_col].map(normalize_ensembl_gene_id)
    normalized["node_type"] = "Gene"
    normalized = normalized.dropna(subset=["node_id"])
    return stable_drop_duplicates(normalized, ["node_id", "node_type"])
