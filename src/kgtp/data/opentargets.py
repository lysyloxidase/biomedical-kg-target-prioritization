"""Open Targets ingestion helpers for OA disease-gene associations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from kgtp.data.common import (
    PathLike,
    coerce_numeric,
    first_present_column,
    normalize_ensembl_gene_id,
    read_table,
    stable_drop_duplicates,
)

OA_DISEASE_EFO = "EFO_0004616"
DEFAULT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

ASSOCIATED_TARGETS_COUNT_QUERY = """
query DiseaseAssociatedTargetsCount($efoId: String!) {
  disease(efoId: $efoId) {
    id
    associatedTargets(page: {index: 0, size: 1}) {
      count
    }
  }
}
"""


def fetch_oa_target_count(
    graphql_url: str = DEFAULT_GRAPHQL_URL,
    disease_efo: str = OA_DISEASE_EFO,
    *,
    timeout_seconds: int = 30,
) -> int:
    """Read the current OA associated-target count from Open Targets GraphQL."""

    payload = {
        "query": ASSOCIATED_TARGETS_COUNT_QUERY,
        "variables": {"efoId": disease_efo},
    }
    response = requests.post(graphql_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    if data.get("errors"):
        msg = f"Open Targets GraphQL errors: {data['errors']}"
        raise RuntimeError(msg)
    disease = data.get("data", {}).get("disease")
    if disease is None:
        msg = f"Disease {disease_efo} was not returned by Open Targets"
        raise ValueError(msg)
    return int(disease["associatedTargets"]["count"])


def normalize_disease_gene_associations(
    associations: pd.DataFrame | PathLike,
    *,
    disease_efo: str = OA_DISEASE_EFO,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Normalize Open Targets disease-gene association rows."""

    df = _as_dataframe(associations)
    disease_col = first_present_column(df, ("disease_id", "diseaseId", "disease"))
    gene_col = first_present_column(df, ("target_id", "targetId", "gene_id", "target"))
    score_col = first_present_column(
        df, ("score", "association_score", "overall_score")
    )

    normalized = pd.DataFrame(
        {
            "disease_id": df[disease_col].astype(str),
            "gene_id": df[gene_col].map(normalize_ensembl_gene_id),
            "score": coerce_numeric(df[score_col]),
            "source": "Open Targets",
        }
    )
    normalized = normalized[
        (normalized["disease_id"] == disease_efo)
        & normalized["gene_id"].notna()
        & (normalized["score"] >= min_score)
    ]
    return stable_drop_duplicates(normalized, ["disease_id", "gene_id"])


def extract_seed_genes(
    associations: pd.DataFrame,
    *,
    max_genes: int | None = None,
) -> pd.Series:
    """Return seed Ensembl genes ordered by descending association score."""

    ordered = associations.sort_values(["score", "gene_id"], ascending=[False, True])
    genes = ordered["gene_id"].drop_duplicates()
    if max_genes is not None:
        genes = genes.head(max_genes)
    return genes.reset_index(drop=True)


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(value)
