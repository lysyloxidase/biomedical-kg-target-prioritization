from __future__ import annotations

import pandas as pd

from kgtp.data.build_graph import (
    assemble_canonical_edges,
    build_node_table,
    check_node_range,
    validate_canonical_edge_types,
)
from kgtp.kg.statistics import (
    GraphStatistics,
    degree_distribution,
    graph_statistics_table,
    known_gene_presence,
)


def _edge_inputs() -> dict[str, pd.DataFrame]:
    return {
        "disease_gene": pd.DataFrame(
            {
                "disease_id": ["EFO_0004616", "EFO_0004616"],
                "gene_id": ["ENSG00000123456", "ENSG00000111111"],
                "score": [0.91, 0.83],
            }
        ),
        "gene_gene": pd.DataFrame(
            {
                "gene_a": ["ENSG00000123456", "ENSG00000111111"],
                "gene_b": ["ENSG00000199999", "ENSG00000199999"],
                "score": [910, 780],
            }
        ),
        "gene_pathway": pd.DataFrame(
            {
                "gene_id": ["ENSG00000123456", "ENSG00000111111"],
                "pathway_id": ["R-HSA-111", "R-HSA-222"],
                "pathway_name": ["Cartilage ECM", "Wnt signaling"],
            }
        ),
        "drug_gene": pd.DataFrame(
            {
                "drug_id": ["CHEMBL1"],
                "gene_id": ["ENSG00000111111"],
                "target_chembl_id": ["CHEMBL_T1"],
                "action_type": ["INHIBITOR"],
            }
        ),
        "gene_go": pd.DataFrame(
            {
                "gene_id": ["ENSG00000123456", "ENSG00000111111"],
                "go_id": ["GO:0001501", "GO:0030198"],
                "evidence_code": ["IDA", "IMP"],
            }
        ),
    }


def test_assemble_edges_and_statistics_cover_phase_1_gates() -> None:
    inputs = _edge_inputs()
    edge_tables = assemble_canonical_edges(**inputs)
    errors = validate_canonical_edge_types(edge_tables)
    attributes = pd.DataFrame(
        {
            "node_id": ["ENSG00000123456", "ENSG00000111111"],
            "node_type": ["Gene", "Gene"],
            "symbol": ["GDF5", "MMP13"],
        }
    )
    nodes = build_node_table(edge_tables, attributes=attributes)

    stats = GraphStatistics.from_tables(nodes, edge_tables)
    presence = known_gene_presence(nodes, ["GDF5", "MMP13", "FRZB"])

    assert errors == []
    assert stats.node_counts["Disease"] == 1
    assert stats.edge_counts["disease_gene"] == 2
    assert stats.positive_disease_gene_links == 2
    assert stats.density > 0
    assert stats.mean_degree > 0
    assert check_node_range(nodes, min_nodes=1, max_nodes=20)
    assert presence == {"FRZB": False, "GDF5": True, "MMP13": True}


def test_degree_distribution_and_statistics_table_are_long_form() -> None:
    edge_tables = assemble_canonical_edges(**_edge_inputs())
    nodes = build_node_table(edge_tables)

    degrees = degree_distribution(edge_tables)
    table = graph_statistics_table(nodes, edge_tables)

    assert degrees.iloc[0]["degree"] >= 2
    assert {"metric", "name", "value"}.issubset(table.columns)
    assert "positive_disease_gene_links" in set(table["metric"])
