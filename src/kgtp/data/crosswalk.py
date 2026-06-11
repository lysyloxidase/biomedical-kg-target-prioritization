"""Normalize source identifiers to an Ensembl gene hub."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kgtp.data.common import (
    PathLike,
    first_present_column,
    normalize_ensembl_gene_id,
    read_table,
    split_identifiers,
    stable_drop_duplicates,
    write_table,
)

STRING_ID_COLUMNS = (
    "string_protein_id",
    "protein_external_id",
    "protein",
    "protein1",
    "string_id",
)
UNIPROT_ID_COLUMNS = ("uniprot_id", "UniProtKB-AC", "From", "Entry", "accession")
ENSEMBL_COLUMNS = ("ensembl_gene_id", "Ensembl", "To", "Gene stable ID", "gene_id")
CHEMBL_TARGET_COLUMNS = ("target_chembl_id", "target_id", "chembl_target_id")


class EnsemblCrosswalk:
    """Build source-to-Ensembl mappings and audit coverage."""

    def __init__(self, min_seed_coverage: float = 0.90) -> None:
        self.min_seed_coverage = min_seed_coverage

    def map_string_to_ensembl(
        self,
        protein_info: pd.DataFrame | PathLike,
        *,
        id_column: str | None = None,
        ensembl_column: str | None = None,
    ) -> pd.DataFrame:
        """Map STRING protein IDs to Ensembl gene IDs.

        The preferred input is a pre-resolved STRING alias/info table containing
        a STRING protein identifier and an Ensembl gene column. If the chosen
        Ensembl column contains version suffixes, they are removed.
        """

        df = _as_dataframe(protein_info)
        source_col = id_column or first_present_column(df, STRING_ID_COLUMNS)
        target_col = ensembl_column or first_present_column(df, ENSEMBL_COLUMNS)

        mapped = pd.DataFrame(
            {
                "source_namespace": "STRING",
                "source_id": df[source_col].astype(str),
                "ensembl_gene_id": df[target_col].map(normalize_ensembl_gene_id),
            }
        )
        mapped = mapped.dropna(subset=["ensembl_gene_id"])
        return stable_drop_duplicates(
            mapped, ["source_namespace", "source_id", "ensembl_gene_id"]
        )

    def map_uniprot_to_ensembl(
        self,
        id_mapping: pd.DataFrame | PathLike,
        *,
        uniprot_column: str | None = None,
        ensembl_column: str | None = None,
    ) -> pd.DataFrame:
        """Map UniProt accessions to Ensembl gene IDs."""

        df = _as_dataframe(id_mapping)
        source_col = uniprot_column or first_present_column(df, UNIPROT_ID_COLUMNS)
        target_col = ensembl_column or first_present_column(df, ENSEMBL_COLUMNS)

        rows: list[dict[str, str]] = []
        for source_id, target_value in zip(
            df[source_col], df[target_col], strict=False
        ):
            for token in split_identifiers(target_value):
                ensembl_gene_id = normalize_ensembl_gene_id(token)
                if ensembl_gene_id is None:
                    continue
                rows.append(
                    {
                        "source_namespace": "UniProt",
                        "source_id": str(source_id),
                        "ensembl_gene_id": ensembl_gene_id,
                    }
                )
        mapped = pd.DataFrame(
            rows, columns=["source_namespace", "source_id", "ensembl_gene_id"]
        )
        return stable_drop_duplicates(
            mapped, ["source_namespace", "source_id", "ensembl_gene_id"]
        )

    def map_chembl_target_to_ensembl(
        self,
        chembl_targets: pd.DataFrame | PathLike,
        uniprot_to_ensembl: pd.DataFrame,
        *,
        chembl_target_column: str | None = None,
        uniprot_column: str | None = None,
    ) -> pd.DataFrame:
        """Map ChEMBL targets through UniProt accessions to Ensembl gene IDs."""

        targets = _as_dataframe(chembl_targets)
        target_col = chembl_target_column or first_present_column(
            targets, CHEMBL_TARGET_COLUMNS
        )
        uniprot_col = uniprot_column or first_present_column(
            targets, UNIPROT_ID_COLUMNS
        )

        uni = uniprot_to_ensembl.rename(
            columns={"source_id": "uniprot_id", "source_namespace": "namespace"}
        )
        merged = targets[[target_col, uniprot_col]].merge(
            uni[["uniprot_id", "ensembl_gene_id"]],
            left_on=uniprot_col,
            right_on="uniprot_id",
            how="left",
        )
        mapped = pd.DataFrame(
            {
                "source_namespace": "ChEMBL",
                "source_id": merged[target_col].astype(str),
                "ensembl_gene_id": merged["ensembl_gene_id"],
            }
        ).dropna(subset=["ensembl_gene_id"])
        return stable_drop_duplicates(
            mapped, ["source_namespace", "source_id", "ensembl_gene_id"]
        )

    def coverage_report(
        self,
        seed_genes: pd.Series | list[str] | set[str],
        mapped_ensembl_ids: pd.Series | list[str] | set[str],
        *,
        required: float | None = None,
    ) -> dict[str, object]:
        """Return seed-gene mapping coverage and pass/fail state."""

        required_coverage = self.min_seed_coverage if required is None else required
        seeds = {
            gene
            for gene in (normalize_ensembl_gene_id(value) for value in seed_genes)
            if gene is not None
        }
        mapped = {
            gene
            for gene in (
                normalize_ensembl_gene_id(value) for value in mapped_ensembl_ids
            )
            if gene is not None
        }
        missing = sorted(seeds - mapped)
        total = len(seeds)
        mapped_count = total - len(missing)
        coverage = mapped_count / total if total else 0.0
        return {
            "seed_gene_count": total,
            "mapped_seed_gene_count": mapped_count,
            "coverage": coverage,
            "required_coverage": required_coverage,
            "passes": coverage > required_coverage,
            "unmapped_seed_genes": missing,
        }

    def write_unmapped_log(
        self,
        seed_genes: pd.Series | list[str] | set[str],
        mapped_ensembl_ids: pd.Series | list[str] | set[str],
        path: PathLike,
    ) -> pd.DataFrame:
        """Write a CSV/TSV/Parquet log of unmapped seed genes."""

        report = self.coverage_report(seed_genes, mapped_ensembl_ids)
        unmapped = pd.DataFrame(
            {
                "ensembl_gene_id": report["unmapped_seed_genes"],
                "reason": "missing_from_crosswalk",
            }
        )
        write_table(unmapped, path)
        return unmapped

    @staticmethod
    def combine(*mappings: pd.DataFrame) -> pd.DataFrame:
        """Combine source mappings into one deduplicated crosswalk table."""

        if not mappings:
            return pd.DataFrame(
                columns=["source_namespace", "source_id", "ensembl_gene_id"]
            )
        combined = pd.concat(mappings, ignore_index=True)
        combined = combined.dropna(subset=["source_id", "ensembl_gene_id"])
        return stable_drop_duplicates(
            combined, ["source_namespace", "source_id", "ensembl_gene_id"]
        )


def _as_dataframe(value: pd.DataFrame | PathLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return read_table(Path(value))
