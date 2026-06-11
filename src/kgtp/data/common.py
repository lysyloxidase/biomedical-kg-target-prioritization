"""Shared dataframe helpers for source ingestion."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

PathLike = str | Path

ENSEMBL_GENE_RE = re.compile(r"(ENSG\d{11})(?:\.\d+)?")


def read_table(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read CSV, TSV, or Parquet data with format inferred from the suffix."""

    path = Path(path)
    suffixes = "".join(path.suffixes[-2:])
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffixes.endswith(".tsv.gz") or path.suffix in {".tsv", ".tab"}:
        return cast(pd.DataFrame, pd.read_csv(path, sep="\t", **kwargs))
    return cast(pd.DataFrame, pd.read_csv(path, **kwargs))


def write_table(df: pd.DataFrame, path: PathLike) -> None:
    """Write a dataframe as Parquet, TSV, or CSV based on suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix in {".tsv", ".tab"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def first_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first candidate column present in `df`."""

    for column in candidates:
        if column in df.columns:
            return column
    available = ", ".join(map(str, df.columns))
    expected = ", ".join(candidates)
    msg = f"None of [{expected}] found. Available columns: {available}"
    raise KeyError(msg)


def optional_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first candidate column present in `df`, if any."""

    for column in candidates:
        if column in df.columns:
            return column
    return None


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required columns: {', '.join(missing)}"
        raise KeyError(msg)


def normalize_ensembl_gene_id(value: object) -> str | None:
    """Normalize a value to a versionless Ensembl gene ID."""

    if _is_missing(value):
        return None
    match = ENSEMBL_GENE_RE.search(str(value))
    if match is None:
        return None
    return match.group(1)


def normalize_ensembl_series(series: pd.Series) -> pd.Series:
    """Normalize all values in a series to versionless Ensembl gene IDs."""

    return series.map(normalize_ensembl_gene_id)


def split_identifiers(value: object) -> list[str]:
    """Split common multi-ID delimiters into cleaned tokens."""

    if _is_missing(value):
        return []
    text = str(value).replace("|", ";").replace(",", ";")
    return [token.strip() for token in text.split(";") if token.strip()]


def coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Coerce a series to numeric values with missing values filled."""

    return pd.to_numeric(series, errors="coerce").fillna(default)


def stable_drop_duplicates(df: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    """Drop duplicate rows while preserving deterministic order."""

    return (
        df.drop_duplicates(subset=list(subset))
        .sort_values(list(subset))
        .reset_index(drop=True)
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return bool(pd.isna(cast(Any, value)))
