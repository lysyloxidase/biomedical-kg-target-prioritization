from __future__ import annotations

import pandas as pd

from kgtp.data.crosswalk import EnsemblCrosswalk


def test_uniprot_crosswalk_splits_and_normalizes_ensembl_ids() -> None:
    mapper = EnsemblCrosswalk(min_seed_coverage=0.90)
    raw = pd.DataFrame(
        {
            "From": ["P11111", "P22222", "P33333"],
            "To": [
                "ENSG00000123456.7; ENSG00000123457",
                "not-an-ensembl-id",
                "ENSG00000123458",
            ],
        }
    )

    mapped = mapper.map_uniprot_to_ensembl(raw)

    assert set(mapped["ensembl_gene_id"]) == {
        "ENSG00000123456",
        "ENSG00000123457",
        "ENSG00000123458",
    }
    assert "ENSG00000123456.7" not in set(mapped["ensembl_gene_id"])


def test_string_and_chembl_crosswalks_map_through_ensembl() -> None:
    mapper = EnsemblCrosswalk()
    string_info = pd.DataFrame(
        {
            "protein_external_id": ["9606.ENSP1", "9606.ENSP2"],
            "ensembl_gene_id": ["ENSG00000111111.1", "ENSG00000122222"],
        }
    )
    uniprot = pd.DataFrame(
        {
            "From": ["P11111", "P22222"],
            "To": ["ENSG00000111111", "ENSG00000133333"],
        }
    )
    chembl_targets = pd.DataFrame(
        {
            "target_chembl_id": ["CHEMBL_T1", "CHEMBL_T2"],
            "uniprot_id": ["P11111", "P22222"],
        }
    )

    string_map = mapper.map_string_to_ensembl(string_info)
    uniprot_map = mapper.map_uniprot_to_ensembl(uniprot)
    chembl_map = mapper.map_chembl_target_to_ensembl(chembl_targets, uniprot_map)

    assert ("STRING", "9606.ENSP1", "ENSG00000111111") in set(
        string_map.itertuples(index=False, name=None)
    )
    assert ("ChEMBL", "CHEMBL_T2", "ENSG00000133333") in set(
        chembl_map.itertuples(index=False, name=None)
    )


def test_coverage_report_requires_strictly_more_than_threshold(tmp_path) -> None:
    mapper = EnsemblCrosswalk(min_seed_coverage=0.90)
    seeds = [
        "ENSG00000100001",
        "ENSG00000100002",
        "ENSG00000100003",
        "ENSG00000100004",
        "ENSG00000100005",
        "ENSG00000100006",
        "ENSG00000100007",
        "ENSG00000100008",
        "ENSG00000100009",
        "ENSG00000100010",
    ]
    mapped = seeds[:9]

    report = mapper.coverage_report(seeds, mapped)
    unmapped = mapper.write_unmapped_log(seeds, mapped, tmp_path / "unmapped.csv")

    assert report["coverage"] == 0.9
    assert report["passes"] is False
    assert unmapped["ensembl_gene_id"].tolist() == ["ENSG00000100010"]
    assert (tmp_path / "unmapped.csv").exists()


def test_combined_crosswalk_deduplicates_without_losing_namespaces() -> None:
    first = pd.DataFrame(
        {
            "source_namespace": ["UniProt", "UniProt"],
            "source_id": ["P11111", "P11111"],
            "ensembl_gene_id": ["ENSG00000111111", "ENSG00000111111"],
        }
    )
    second = pd.DataFrame(
        {
            "source_namespace": ["ChEMBL"],
            "source_id": ["CHEMBL_T1"],
            "ensembl_gene_id": ["ENSG00000111111"],
        }
    )

    combined = EnsemblCrosswalk.combine(first, second)

    assert len(combined) == 2
    assert set(combined["source_namespace"]) == {"UniProt", "ChEMBL"}
