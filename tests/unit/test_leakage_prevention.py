from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from kgtp.data.build_graph import assemble_canonical_edges
from kgtp.hetero.auxiliary_splits import (
    attach_auxiliary_supervision,
    split_auxiliary_relations,
)
from kgtp.hetero.build_heterodata import build_heterodata
from kgtp.hetero.feature_transformers import TrainGraphFeatureTransformer
from kgtp.hetero.split_protocol import (
    TargetSplit,
    build_split_bundle,
    hash_graph,
    split_target_relation,
    validate_target_split,
)

ROOT = Path(__file__).resolve().parents[2]
EDGE_TYPE = ("disease", "associated_with", "gene")
REVERSE_EDGE_TYPE = ("gene", "rev_associated_with", "disease")


def _sample_graph() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sample = ROOT / "data" / "sample"
    nodes = pd.read_parquet(sample / "nodes.parquet")
    edges = assemble_canonical_edges(
        disease_gene=pd.read_parquet(sample / "disease_gene.parquet"),
        gene_gene=pd.read_parquet(sample / "gene_gene.parquet"),
        gene_pathway=pd.read_parquet(sample / "gene_pathway.parquet"),
        drug_gene=pd.read_parquet(sample / "drug_gene.parquet"),
        gene_go=pd.read_parquet(sample / "gene_go.parquet"),
        pathway_pathway=pd.read_parquet(sample / "pathway_pathway.parquet"),
    )
    return nodes, edges


def _split_and_train_graph(
    seed: int = 13,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    TargetSplit,
    dict[str, pd.DataFrame],
]:
    nodes, reference_edges = _sample_graph()
    split = split_target_relation(nodes, reference_edges, seed=seed)
    train_edges = dict(reference_edges)
    train_edges["disease_gene"] = split.message_edges
    return nodes, reference_edges, split, train_edges


def _fitted_features(
    nodes: pd.DataFrame,
    train_edges: dict[str, pd.DataFrame],
) -> tuple[TrainGraphFeatureTransformer, dict[str, torch.Tensor]]:
    graph_hash = hash_graph(nodes, train_edges)
    transformer = TrainGraphFeatureTransformer().fit(
        nodes, train_edges, graph_hash=graph_hash
    )
    return transformer, transformer.transform(nodes)


@pytest.fixture
def corrupted_split_fixture() -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    TargetSplit,
]:
    nodes, edges = _sample_graph()
    split = split_target_relation(nodes, edges, seed=13)
    duplicate = split.assignments.iloc[[0]]
    corrupted = replace(
        split,
        assignments=pd.concat([split.assignments, duplicate], ignore_index=True),
    )
    return nodes, edges, corrupted


def _perturb_partition(
    reference_edges: dict[str, pd.DataFrame],
    split: TargetSplit,
    partition: str,
) -> dict[str, pd.DataFrame]:
    held_out = split.assignments.loc[
        split.assignments["partition"] == partition,
        ["source_id", "target_id"],
    ].iloc[0]
    perturbed = {name: frame.copy() for name, frame in reference_edges.items()}
    disease_gene = perturbed["disease_gene"]
    mask = (disease_gene["source_id"] == held_out["source_id"]) & (
        disease_gene["target_id"] == held_out["target_id"]
    )
    perturbed["disease_gene"] = disease_gene.loc[~mask].reset_index(drop=True)
    return perturbed


def test_test_edges_do_not_change_training_features() -> None:
    nodes, reference_edges, split, train_edges = _split_and_train_graph()
    _, first = _fitted_features(nodes, train_edges)
    perturbed_reference = _perturb_partition(reference_edges, split, "test")
    second_train_edges = dict(perturbed_reference)
    second_train_edges["disease_gene"] = split.message_edges
    _, second = _fitted_features(nodes, second_train_edges)

    assert torch.equal(first["gene"], second["gene"])
    assert torch.equal(first["pathway"], second["pathway"])
    assert torch.equal(first["go_term"], second["go_term"])


def test_validation_edges_do_not_change_training_features() -> None:
    nodes, reference_edges, split, train_edges = _split_and_train_graph()
    _, first = _fitted_features(nodes, train_edges)
    perturbed_reference = _perturb_partition(reference_edges, split, "validation")
    second_train_edges = dict(perturbed_reference)
    second_train_edges["disease_gene"] = split.message_edges
    _, second = _fitted_features(nodes, second_train_edges)

    assert torch.equal(first["gene"], second["gene"])


def test_test_edges_do_not_change_training_pagerank() -> None:
    nodes, reference_edges, split, train_edges = _split_and_train_graph()
    _, first = _fitted_features(nodes, train_edges)
    perturbed_reference = _perturb_partition(reference_edges, split, "test")
    second_train_edges = dict(perturbed_reference)
    second_train_edges["disease_gene"] = split.message_edges
    _, second = _fitted_features(nodes, second_train_edges)

    assert torch.equal(first["gene"][:, 1], second["gene"][:, 1])


def test_reverse_test_edges_are_absent_from_message_graph() -> None:
    nodes, _, split, train_edges = _split_and_train_graph()
    _, features = _fitted_features(nodes, train_edges)
    data = build_heterodata(
        nodes,
        train_edges,
        gene_feature_mode="none",
        precomputed_features=features,
    )
    bundle = build_split_bundle(
        data,
        split,
        edge_type=EDGE_TYPE,
        reverse_edge_type=REVERSE_EDGE_TYPE,
    )
    test_labels = bundle.test_data[EDGE_TYPE].edge_label
    test_edges = bundle.test_data[EDGE_TYPE].edge_label_index[:, test_labels == 1]
    reverse_test = {
        (int(target), int(source)) for source, target in test_edges.t().tolist()
    }
    for view in (bundle.train_data, bundle.val_data, bundle.test_data):
        reverse_message = {
            (int(source), int(target))
            for source, target in view[REVERSE_EDGE_TYPE].edge_index.t().tolist()
        }
        assert reverse_test.isdisjoint(reverse_message)


def test_train_supervision_edges_are_absent_from_message_graph() -> None:
    nodes, _, split, train_edges = _split_and_train_graph()
    _, features = _fitted_features(nodes, train_edges)
    data = build_heterodata(
        nodes,
        train_edges,
        gene_feature_mode="none",
        precomputed_features=features,
    )
    bundle = build_split_bundle(
        data,
        split,
        edge_type=EDGE_TYPE,
        reverse_edge_type=REVERSE_EDGE_TYPE,
    )
    labels = bundle.train_data[EDGE_TYPE].edge_label
    train_supervision = {
        (int(source), int(target))
        for source, target in bundle.train_data[EDGE_TYPE]
        .edge_label_index[:, labels == 1]
        .t()
        .tolist()
    }
    message_edges = {
        (int(source), int(target))
        for source, target in bundle.train_data[EDGE_TYPE].edge_index.t().tolist()
    }
    assert train_supervision.isdisjoint(message_edges)


def test_feature_transformers_fit_on_train_only() -> None:
    nodes, reference_edges, split, train_edges = _split_and_train_graph()
    transformer, first = _fitted_features(nodes, train_edges)

    assert transformer.fitted_graph_hash == split.metadata["train_message_graph_hash"]
    assert transformer.fitted_graph_hash != hash_graph(nodes, reference_edges)
    assert transformer.to_dict()["fit_scope"] == "train_message_graph_only"

    mutated_reference = _perturb_partition(reference_edges, split, "test")
    del mutated_reference
    second = transformer.transform(nodes)
    assert torch.equal(first["gene"], second["gene"])


def test_split_is_deterministic_for_same_seed() -> None:
    nodes, edges = _sample_graph()
    first = split_target_relation(nodes, edges, seed=29)
    second = split_target_relation(nodes, edges, seed=29)

    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.train_supervision, second.train_supervision)
    assert first.metadata == second.metadata


def test_split_changes_for_different_seed() -> None:
    nodes, edges = _sample_graph()
    first = split_target_relation(nodes, edges, seed=13)
    second = split_target_relation(nodes, edges, seed=17)

    assert not first.assignments.equals(second.assignments)


def test_corrupted_split_is_rejected(
    corrupted_split_fixture: tuple[
        pd.DataFrame,
        dict[str, pd.DataFrame],
        TargetSplit,
    ],
) -> None:
    nodes, edges, corrupted = corrupted_split_fixture

    with pytest.raises(ValueError, match="assigned to more than one partition"):
        validate_target_split(corrupted, nodes=nodes, edge_tables=edges)


def test_duplicate_reference_target_edge_is_rejected() -> None:
    nodes, edges = _sample_graph()
    corrupted_edges = dict(edges)
    corrupted_edges["disease_gene"] = pd.concat(
        [edges["disease_gene"], edges["disease_gene"].iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate canonical edges"):
        split_target_relation(nodes, corrupted_edges, seed=13)


def test_auxiliary_supervision_and_reverse_edges_are_absent_from_message_graph() -> (
    None
):
    nodes, reference_edges = _sample_graph()
    target_split = split_target_relation(nodes, reference_edges, seed=13)
    auxiliary, train_edges = split_auxiliary_relations(
        nodes,
        reference_edges,
        seed=13,
    )
    train_edges["disease_gene"] = target_split.message_edges
    _, features = _fitted_features(nodes, train_edges)
    data = build_heterodata(
        nodes,
        train_edges,
        gene_feature_mode="none",
        precomputed_features=features,
    )
    bundle = attach_auxiliary_supervision(
        build_split_bundle(
            data,
            target_split,
            edge_type=EDGE_TYPE,
            reverse_edge_type=REVERSE_EDGE_TYPE,
        ),
        auxiliary,
    )

    for task in auxiliary.values():
        for view in (bundle.train_data, bundle.val_data, bundle.test_data):
            labels = view[task.spec.edge_type].edge_label
            positives = {
                (int(source), int(target))
                for source, target in view[task.spec.edge_type]
                .edge_label_index[:, labels == 1]
                .t()
                .tolist()
            }
            message = {
                (int(source), int(target))
                for source, target in view[task.spec.edge_type].edge_index.t().tolist()
            }
            reverse = {
                (int(source), int(target))
                for source, target in view[task.spec.reverse_edge_type]
                .edge_index.t()
                .tolist()
            }
            assert positives.isdisjoint(message)
            assert {(target, source) for source, target in positives}.isdisjoint(
                reverse
            )
