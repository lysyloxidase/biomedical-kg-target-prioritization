"""Reactome gene-pathway ingestion helpers."""

from __future__ import annotations

import pandas as pd

from kgtp.data.common import (
    PathLike,
    first_present_column,
    normalize_ensembl_gene_id,
    read_table,
    stable_drop_duplicates,
)


def read_ensembl2reactome(path: PathLike) -> pd.DataFrame:
    """Read the Reactome Ensembl2Reactome bulk mapping file."""

    columns = [
        "gene_id",
        "pathway_id",
        "url",
        "pathway_name",
        "evidence_code",
        "species",
    ]
    return pd.read_csv(path, sep="\t", names=columns, comment="#")


def normalize_gene_pathway_edges(
    mappings: pd.DataFrame | PathLike,
    *,
    species: str = "Homo sapiens",
) -> pd.DataFrame:
    """Normalize Reactome gene-pathway mappings."""

    df = _as_dataframe(mappings)
    gene_col = first_present_column(df, ("gene_id", "ensembl_gene_id", "source_id"))
    pathway_col = first_present_column(df, ("pathway_id", "reactome_id", "target_id"))
    species_col = "species" if "species" in df.columns else None

    normalized = pd.DataFrame(
        {
            "gene_id": df[gene_col].map(normalize_ensembl_gene_id),
            "pathway_id": df[pathway_col].astype(str),
            "source": "Reactome",
        }
    )
    if "pathway_name" in df.columns:
        normalized["pathway_name"] = df["pathway_name"].astype(str)
    if species_col is not None:
        normalized = normalized[df[species_col].astype(str) == species]
    normalized = normalized.dropna(subset=["gene_id"])
    normalized = normalized[normalized["pathway_id"].str.startswith("R-HSA")]
    return stable_drop_duplicates(normalized, ["gene_id", "pathway_id"])


def normalize_pathway_hierarchy(hierarchy: pd.DataFrame | PathLike) -> pd.DataFrame:
    """Normalize optional Reactome pathway hierarchy edges."""

    df = _as_dataframe(hierarchy)
    parent_col = first_present_column(df, ("parent_id", "parent", "source_id"))
    child_col = first_present_column(df, ("child_id", "child", "target_id"))
    normalized = pd.DataFrame(
        {
            "parent_pathway_id": df[parent_col].astype(str),
            "child_pathway_id": df[child_col].astype(str),
            "source": "Reactome",
        }
    )
    normalized = normalized[
        normalized["parent_pathway_id"].str.startswith("R-HSA")
        & normalized["child_pathway_id"].str.startswith("R-HSA")
    ]
    return stable_drop_duplicates(normalized, ["parent_pathway_id", "child_pathway_id"])


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(value)
