"""STRING PPI normalization and OA-neighborhood expansion."""

from __future__ import annotations

import pandas as pd

from kgtp.data.common import (
    PathLike,
    coerce_numeric,
    first_present_column,
    normalize_ensembl_gene_id,
    read_table,
    stable_drop_duplicates,
)


def filter_high_confidence(
    links: pd.DataFrame | PathLike,
    *,
    threshold: int = 700,
) -> pd.DataFrame:
    """Keep STRING links at or above the configured combined score threshold."""

    df = _as_dataframe(links)
    p1 = first_present_column(df, ("protein1", "source_id", "gene_a"))
    p2 = first_present_column(df, ("protein2", "target_id", "gene_b"))
    score = first_present_column(df, ("combined_score", "score"))

    filtered = pd.DataFrame(
        {
            "string_protein_a": df[p1].astype(str),
            "string_protein_b": df[p2].astype(str),
            "score": coerce_numeric(df[score]),
        }
    )
    filtered = filtered[filtered["score"] >= threshold]
    return stable_drop_duplicates(filtered, ["string_protein_a", "string_protein_b"])


def map_string_ppi_to_genes(
    ppi_edges: pd.DataFrame,
    string_to_ensembl: pd.DataFrame,
) -> pd.DataFrame:
    """Convert STRING protein-protein edges into Ensembl gene-gene edges."""

    mapping = string_to_ensembl[["source_id", "ensembl_gene_id"]]
    left = ppi_edges.merge(
        mapping.rename(
            columns={"source_id": "string_protein_a", "ensembl_gene_id": "gene_a"}
        ),
        on="string_protein_a",
        how="left",
    )
    both = left.merge(
        mapping.rename(
            columns={"source_id": "string_protein_b", "ensembl_gene_id": "gene_b"}
        ),
        on="string_protein_b",
        how="left",
    )
    mapped = both.dropna(subset=["gene_a", "gene_b"])
    mapped = mapped[mapped["gene_a"] != mapped["gene_b"]]
    return stable_drop_duplicates(
        mapped[["gene_a", "gene_b", "score"]], ["gene_a", "gene_b"]
    )


def normalize_gene_ppi(
    links: pd.DataFrame | PathLike, *, threshold: int = 700
) -> pd.DataFrame:
    """Normalize already gene-keyed STRING-like PPI tables."""

    df = _as_dataframe(links)
    source_col = first_present_column(df, ("gene_a", "source_id", "protein1"))
    target_col = first_present_column(df, ("gene_b", "target_id", "protein2"))
    score_col = first_present_column(df, ("score", "combined_score"))
    normalized = pd.DataFrame(
        {
            "gene_a": df[source_col].map(normalize_ensembl_gene_id),
            "gene_b": df[target_col].map(normalize_ensembl_gene_id),
            "score": coerce_numeric(df[score_col]),
        }
    )
    normalized = normalized.dropna(subset=["gene_a", "gene_b"])
    normalized = normalized[
        (normalized["gene_a"] != normalized["gene_b"])
        & (normalized["score"] >= threshold)
    ]
    return stable_drop_duplicates(normalized, ["gene_a", "gene_b"])


def one_hop_expand(
    seed_genes: set[str] | list[str] | pd.Series,
    ppi_edges: pd.DataFrame,
    *,
    max_total_genes: int | None = None,
) -> pd.DataFrame:
    """Return PPI edges touching at least one seed gene, with optional gene cap."""

    seed_set = {gene for gene in seed_genes if isinstance(gene, str)}
    expanded = ppi_edges[
        ppi_edges["gene_a"].isin(seed_set) | ppi_edges["gene_b"].isin(seed_set)
    ].copy()
    expanded = expanded.sort_values(
        ["score", "gene_a", "gene_b"], ascending=[False, True, True]
    )
    if max_total_genes is None:
        return expanded.reset_index(drop=True)

    kept_genes = set(seed_set)
    keep_mask: list[bool] = []
    for _, row in expanded.iterrows():
        candidates = {str(row["gene_a"]), str(row["gene_b"])}
        if len(kept_genes | candidates) <= max_total_genes:
            kept_genes.update(candidates)
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return expanded.loc[keep_mask].reset_index(drop=True)


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(value)
