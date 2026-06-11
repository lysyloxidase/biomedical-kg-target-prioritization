"""Gene Ontology Annotation ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kgtp.data.common import (
    PathLike,
    first_present_column,
    read_table,
    stable_drop_duplicates,
)

GAF_COLUMNS = [
    "db",
    "db_object_id",
    "db_object_symbol",
    "qualifier",
    "go_id",
    "db_reference",
    "evidence_code",
    "with_or_from",
    "aspect",
    "db_object_name",
    "db_object_synonym",
    "db_object_type",
    "taxon",
    "date",
    "assigned_by",
    "annotation_extension",
    "gene_product_form_id",
]


def read_gaf(path: PathLike) -> pd.DataFrame:
    """Read a GOA GAF 2.x file."""

    return pd.read_csv(
        Path(path),
        sep="\t",
        comment="!",
        names=GAF_COLUMNS,
        dtype=str,
        low_memory=False,
    )


def normalize_gene_go_edges(
    annotations: pd.DataFrame | PathLike,
    uniprot_to_ensembl: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize GO annotations to Ensembl gene-GO edges."""

    df = _as_dataframe(annotations)
    uniprot_col = first_present_column(
        df, ("uniprot_id", "db_object_id", "UniProtKB-AC")
    )
    go_col = first_present_column(df, ("go_id", "GO_ID", "target_id"))
    evidence_col = "evidence_code" if "evidence_code" in df.columns else None

    merged = df.merge(
        uniprot_to_ensembl[["source_id", "ensembl_gene_id"]],
        left_on=uniprot_col,
        right_on="source_id",
        how="left",
    )
    normalized = pd.DataFrame(
        {
            "gene_id": merged["ensembl_gene_id"],
            "go_id": merged[go_col].astype(str),
            "source": "GOA",
        }
    ).dropna(subset=["gene_id"])
    if evidence_col is not None:
        normalized["evidence_code"] = merged[evidence_col].astype(str)
    normalized = normalized[normalized["go_id"].str.startswith("GO:")]
    return stable_drop_duplicates(normalized, ["gene_id", "go_id"])


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(value)
