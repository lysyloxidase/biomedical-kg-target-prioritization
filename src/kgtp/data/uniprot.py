"""UniProt mapping helpers."""

from __future__ import annotations

import pandas as pd

from kgtp.data.crosswalk import EnsemblCrosswalk


def normalize_uniprot_ensembl_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Normalize a UniProt ID mapping table to the crosswalk schema."""

    return EnsemblCrosswalk().map_uniprot_to_ensembl(mapping)


def normalize_uniprot_go_annotations(
    annotations: pd.DataFrame,
    uniprot_to_ensembl: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize UniProt GO annotations to Ensembl gene IDs."""

    merged = annotations.merge(
        uniprot_to_ensembl[["source_id", "ensembl_gene_id"]],
        left_on="uniprot_id",
        right_on="source_id",
        how="left",
    )
    normalized = merged.dropna(subset=["ensembl_gene_id", "go_id"])
    return normalized.rename(columns={"ensembl_gene_id": "gene_id"})[
        ["gene_id", "go_id", "evidence_code", "source"]
    ].drop_duplicates()
