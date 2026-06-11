"""ChEMBL mechanism-of-action target normalization."""

from __future__ import annotations

import pandas as pd

from kgtp.data.common import (
    PathLike,
    first_present_column,
    read_table,
    stable_drop_duplicates,
)


def normalize_drug_target_edges(
    mechanisms: pd.DataFrame | PathLike,
    chembl_target_to_ensembl: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize ChEMBL MoA rows into drug-target gene edges."""

    df = _as_dataframe(mechanisms)
    drug_col = first_present_column(
        df, ("molecule_chembl_id", "parent_molecule_chembl_id", "drug_id")
    )
    target_col = first_present_column(df, ("target_chembl_id", "target_id"))

    merged = df.merge(
        chembl_target_to_ensembl[["source_id", "ensembl_gene_id"]],
        left_on=target_col,
        right_on="source_id",
        how="left",
    )
    normalized = pd.DataFrame(
        {
            "drug_id": merged[drug_col].astype(str),
            "gene_id": merged["ensembl_gene_id"],
            "target_chembl_id": merged[target_col].astype(str),
            "source": "ChEMBL",
        }
    ).dropna(subset=["gene_id"])
    if "action_type" in merged.columns:
        normalized["action_type"] = merged["action_type"].astype(str)
    if "mechanism_of_action" in merged.columns:
        normalized["mechanism_of_action"] = merged["mechanism_of_action"].astype(str)
    return stable_drop_duplicates(
        normalized, ["drug_id", "gene_id", "target_chembl_id"]
    )


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(value)
