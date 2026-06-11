from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from kgtp.data.build_graph import assemble_canonical_edges, build_node_table
from kgtp.hetero.build_heterodata import build_heterodata
from kgtp.hetero.negatives import edge_index_to_pairs, sample_negative_edges
from kgtp.hetero.splits import (
    DEFAULT_EDGE_TYPES,
    DEFAULT_REV_EDGE_TYPES,
    leakage_free_random_link_split,
    load_splits,
    message_edge_pairs,
    save_splits,
    supervision_edge_pairs,
)


def _phase2_tables() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    disease_gene = pd.DataFrame(
        {
            "disease_id": [
                "EFO_0004616",
                "EFO_0004616",
                "EFO_0004616",
                "EFO_0004617",
                "EFO_0004617",
                "EFO_0004617",
                "EFO_0004618",
                "EFO_0004618",
                "EFO_0004618",
                "EFO_0004618",
            ],
            "gene_id": [f"ENSG0000010000{i}" for i in range(10)],
            "score": [0.95, 0.93, 0.91, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76],
        }
    )
    gene_gene = pd.DataFrame(
        {
            "gene_a": [f"ENSG0000010000{i}" for i in range(8)],
            "gene_b": [f"ENSG0000010000{i}" for i in range(2, 10)],
            "score": [900, 880, 860, 840, 820, 800, 780, 760],
        }
    )
    gene_pathway = pd.DataFrame(
        {
            "gene_id": [f"ENSG0000010000{i}" for i in range(10)],
            "pathway_id": [f"R-HSA-{100 + (i % 5)}" for i in range(10)],
            "pathway_name": ["cartilage", "wnt", "nfkb", "mapk", "ecm"] * 2,
        }
    )
    drug_gene = pd.DataFrame(
        {
            "drug_id": [f"CHEMBL{i}" for i in range(8)],
            "gene_id": [f"ENSG0000010000{i}" for i in range(8)],
            "target_chembl_id": [f"CHEMBL_T{i}" for i in range(8)],
            "action_type": ["INHIBITOR"] * 8,
            "mechanism_of_action": ["test mechanism"] * 8,
        }
    )
    gene_go = pd.DataFrame(
        {
            "gene_id": [f"ENSG0000010000{i}" for i in range(10)],
            "go_id": [f"GO:000000{i % 4}" for i in range(10)],
            "evidence_code": ["IDA", "IMP", "IEA", "TAS", "IDA"] * 2,
        }
    )
    pathway_pathway = pd.DataFrame(
        {
            "parent_pathway_id": ["R-HSA-100", "R-HSA-100", "R-HSA-101"],
            "child_pathway_id": ["R-HSA-101", "R-HSA-102", "R-HSA-103"],
        }
    )

    edge_tables = assemble_canonical_edges(
        disease_gene=disease_gene,
        gene_gene=gene_gene,
        gene_pathway=gene_pathway,
        drug_gene=drug_gene,
        gene_go=gene_go,
        pathway_pathway=pathway_pathway,
    )
    attributes = pd.DataFrame(
        [
            {
                "node_id": "EFO_0004616",
                "node_type": "Disease",
                "label": "knee osteoarthritis",
            },
            {
                "node_id": "EFO_0004617",
                "node_type": "Disease",
                "label": "hip osteoarthritis",
            },
            {
                "node_id": "EFO_0004618",
                "node_type": "Disease",
                "label": "hand osteoarthritis",
            },
            *[
                {
                    "node_id": f"ENSG0000010000{i}",
                    "node_type": "Gene",
                    "symbol": f"GENE{i}",
                }
                for i in range(10)
            ],
            *[
                {
                    "node_id": f"CHEMBL{i}",
                    "node_type": "Drug",
                    "smiles": smiles,
                }
                for i, smiles in enumerate(
                    ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CN", "CO"]
                )
            ],
            *[
                {
                    "node_id": f"GO:000000{i}",
                    "node_type": "GOTerm",
                    "namespace": ["BP", "MF", "CC", "BP"][i],
                    "information_content": 1.0 + i,
                }
                for i in range(4)
            ],
        ]
    )
    nodes = build_node_table(edge_tables, attributes=attributes)
    return nodes, edge_tables


def _phase2_heterodata(tmp_path: Path):
    nodes, edge_tables = _phase2_tables()
    return build_heterodata(
        nodes, edge_tables, output_dir=tmp_path, gene_feature_mode="go"
    )


def test_heterodata_export_has_all_node_types_features_metadata_and_maps(
    tmp_path,
) -> None:
    data = _phase2_heterodata(tmp_path)
    node_types, edge_types = data.metadata()

    assert set(node_types) == {"disease", "gene", "drug", "pathway", "go_term"}
    assert set(DEFAULT_EDGE_TYPES).issubset(set(edge_types))
    assert set(DEFAULT_REV_EDGE_TYPES).issubset(set(edge_types))
    assert data["gene"].x.size(1) >= 2
    assert data["gene"].x[:, 0].sum() > 0
    assert data["drug"].x.shape == (8, 2048)
    assert data["drug"].x.sum() > 0
    assert data["pathway"].x.size(1) == 2
    assert data["go_term"].x.size(1) == 4

    maps_path = tmp_path / "node_index_maps.json"
    assert maps_path.exists()
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    assert maps["gene"]["ENSG00000100000"] == 0


def test_split_leakage_guards_hold_for_supervision_edges(tmp_path) -> None:
    data = _phase2_heterodata(tmp_path)
    bundle = leakage_free_random_link_split(
        data,
        seed=7,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.5,
        train_neg_sampling_ratio=1.0,
        eval_neg_sampling_ratio=2.0,
    )

    for edge_type, rev_edge_type in zip(
        DEFAULT_EDGE_TYPES, DEFAULT_REV_EDGE_TYPES, strict=True
    ):
        train_pos = supervision_edge_pairs(bundle.train_data, edge_type, label=1)
        val_pos = supervision_edge_pairs(bundle.val_data, edge_type, label=1)
        test_pos = supervision_edge_pairs(bundle.test_data, edge_type, label=1)

        assert train_pos.isdisjoint(val_pos)
        assert train_pos.isdisjoint(test_pos)
        assert val_pos.isdisjoint(test_pos)

        reverse_test_pos = {(target, source) for source, target in test_pos}
        assert reverse_test_pos.isdisjoint(
            message_edge_pairs(bundle.train_data, rev_edge_type)
        )
        assert reverse_test_pos.isdisjoint(
            message_edge_pairs(bundle.val_data, rev_edge_type)
        )
        assert reverse_test_pos.isdisjoint(
            message_edge_pairs(bundle.test_data, rev_edge_type)
        )

        train_neg = supervision_edge_pairs(bundle.train_data, edge_type, label=0)
        val_neg = supervision_edge_pairs(bundle.val_data, edge_type, label=0)
        test_neg = supervision_edge_pairs(bundle.test_data, edge_type, label=0)
        assert train_neg.isdisjoint(val_neg)
        assert train_neg.isdisjoint(test_neg)
        assert val_neg.isdisjoint(test_neg)

        assert train_pos.isdisjoint(message_edge_pairs(bundle.train_data, edge_type))

    disease_gene = ("disease", "associated_with", "gene")
    assert supervision_edge_pairs(bundle.train_data, disease_gene, label=1).isdisjoint(
        supervision_edge_pairs(bundle.test_data, disease_gene, label=1)
    )
    assert bundle.metadata["seed"] == 7
    assert bundle.metadata["disjoint_train_ratio"] == 0.5


def test_splits_save_reload_and_reproduce_for_same_seed(tmp_path) -> None:
    data = _phase2_heterodata(tmp_path / "heterodata")
    first = leakage_free_random_link_split(
        data,
        seed=11,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.4,
        eval_neg_sampling_ratio=2.0,
    )
    second = leakage_free_random_link_split(
        data,
        seed=11,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.4,
        eval_neg_sampling_ratio=2.0,
    )

    edge_type = ("disease", "associated_with", "gene")
    assert torch.equal(
        first.train_data[edge_type].edge_label_index,
        second.train_data[edge_type].edge_label_index,
    )

    save_splits(first, tmp_path / "splits")
    loaded = load_splits(tmp_path / "splits")

    assert loaded.metadata == first.metadata
    assert torch.equal(
        loaded.test_data[edge_type].edge_label_index,
        first.test_data[edge_type].edge_label_index,
    )
    assert (tmp_path / "splits" / "split_metadata.json").exists()


def test_negative_sampling_strategies_exclude_positives_and_used_edges() -> None:
    positive = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
    used = {(2, 2)}

    random_neg = sample_negative_edges(
        num_src_nodes=4,
        num_dst_nodes=4,
        positive_edge_index=positive,
        num_samples=4,
        seed=1,
        strategy="random",
        used_edges=used,
    )
    degree_neg = sample_negative_edges(
        num_src_nodes=4,
        num_dst_nodes=4,
        positive_edge_index=positive,
        num_samples=4,
        seed=1,
        strategy="degree_matched",
        source_degrees=torch.tensor([5.0, 4.0, 3.0, 2.0]),
        target_degrees=torch.tensor([5.0, 4.0, 3.0, 2.0]),
        used_edges=used,
    )
    hard_neg = sample_negative_edges(
        num_src_nodes=4,
        num_dst_nodes=4,
        positive_edge_index=positive,
        num_samples=2,
        seed=1,
        strategy="hard",
        hard_candidates_by_source={0: [2, 3], 1: [2]},
        used_edges=used,
    )

    forbidden = edge_index_to_pairs(positive) | used
    assert edge_index_to_pairs(random_neg).isdisjoint(forbidden)
    assert edge_index_to_pairs(degree_neg).isdisjoint(forbidden)
    assert edge_index_to_pairs(hard_neg).isdisjoint(forbidden)
