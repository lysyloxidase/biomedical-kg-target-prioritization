from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from kgtp.hetero.splits import leakage_free_random_link_split
from kgtp.models.decoder import (
    DistMultDecoder,
    DotProductDecoder,
    MLPDecoder,
    make_decoder,
)
from kgtp.models.encoder import HGTConv, InputProjection, RGCNConv, SAGEConv, to_hetero
from kgtp.models.graphsage import GraphSAGEEncoder, HomogeneousGraphSAGEEncoder
from kgtp.models.hgt import HGTEncoder
from kgtp.models.multitask import PREDICTED_EDGE_TYPES, MultiTaskLinkPredictor
from kgtp.models.rgcn import RGCNEncoder
from kgtp.models.train import (
    TrainingConfig,
    build_model,
    evaluate_split,
    hgt_beats_popularity_floor,
    train_multiseed,
    train_one_seed,
)


def _toy_heterodata() -> HeteroData:
    data = HeteroData()
    data["disease"].x = torch.eye(3)
    data["disease"].node_id = ["D1", "D2", "D3"]
    data["gene"].x = torch.randn(8, 5, generator=torch.Generator().manual_seed(1))
    data["gene"].node_id = [f"G{i}" for i in range(8)]
    data["drug"].x = torch.randn(4, 4, generator=torch.Generator().manual_seed(2))
    data["drug"].node_id = [f"DR{i}" for i in range(4)]
    data["pathway"].x = torch.randn(5, 2, generator=torch.Generator().manual_seed(3))
    data["pathway"].node_id = [f"P{i}" for i in range(5)]
    data["go_term"].x = torch.randn(4, 4, generator=torch.Generator().manual_seed(4))
    data["go_term"].node_id = [f"GO:{i}" for i in range(4)]

    disease_gene = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 2, 3, 4, 4, 5, 6]],
        dtype=torch.long,
    )
    drug_gene = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    gene_pathway = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 1, 2, 2, 3, 3, 4]],
        dtype=torch.long,
    )
    gene_gene = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
    gene_go = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 0, 1]], dtype=torch.long)

    data[("disease", "associated_with", "gene")].edge_index = disease_gene
    data[("gene", "rev_associated_with", "disease")].edge_index = disease_gene.flip(0)
    data[("drug", "targets", "gene")].edge_index = drug_gene
    data[("gene", "rev_targets", "drug")].edge_index = drug_gene.flip(0)
    data[("gene", "participates_in", "pathway")].edge_index = gene_pathway
    data[("pathway", "rev_participates_in", "gene")].edge_index = gene_pathway.flip(0)
    data[("gene", "interacts", "gene")].edge_index = gene_gene
    data[("gene", "annotated_with", "go_term")].edge_index = gene_go
    data[("go_term", "rev_annotated_with", "gene")].edge_index = gene_go.flip(0)
    return data


def _split_bundle():
    data = _toy_heterodata()
    bundle = leakage_free_random_link_split(
        data,
        seed=13,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.5,
        train_neg_sampling_ratio=1.0,
        eval_neg_sampling_ratio=1.0,
    )
    return data, bundle


def test_hgt_encoder_consumes_metadata_and_validates_heads() -> None:
    data = _toy_heterodata()
    metadata = data.metadata()
    encoder = HGTEncoder(hidden=8, num_heads=2, num_layers=1, metadata=metadata)
    out = encoder(data.x_dict, data.edge_index_dict)

    assert set(out) == set(data.node_types)
    assert out["gene"].shape == (8, 8)
    with pytest.raises(ValueError, match="divisible"):
        HGTEncoder(hidden=7, num_heads=2, num_layers=1, metadata=metadata)


def test_shared_encoder_exports_and_input_projection() -> None:
    projection = InputProjection(6, ["gene", "disease"])
    out = projection({"gene": torch.ones(3, 4), "disease": torch.ones(2, 2)})

    assert out["gene"].shape == (3, 6)
    assert out["disease"].shape == (2, 6)
    assert HGTConv is not None
    assert RGCNConv is not None
    assert SAGEConv is not None
    assert to_hetero is not None


def test_decoder_options_and_multitask_heads_work() -> None:
    data = _toy_heterodata()
    z_src = torch.randn(4, 8, generator=torch.Generator().manual_seed(5))
    z_dst = torch.randn(5, 8, generator=torch.Generator().manual_seed(6))
    edge_label_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    for name, cls in (
        ("dot", DotProductDecoder),
        ("distmult", DistMultDecoder),
        ("mlp", MLPDecoder),
    ):
        decoder = make_decoder(
            name,  # type: ignore[arg-type]
            hidden_channels=8,
            edge_types=[("drug", "targets", "gene")],
        )
        scores = decoder(z_src, z_dst, edge_label_index, ("drug", "targets", "gene"))
        assert isinstance(decoder, cls)
        assert scores.shape == (3,)

    model = MultiTaskLinkPredictor(
        HGTEncoder(8, 2, 1, data.metadata()),
        hidden_channels=8,
        decoder_name="dot",
    )
    outputs = model(
        data.x_dict,
        data.edge_index_dict,
        {
            edge_type: data[edge_type].edge_index[:, :2]
            for edge_type in PREDICTED_EDGE_TYPES
        },
    )
    assert set(outputs) == set(PREDICTED_EDGE_TYPES)
    assert all(value.numel() == 2 for value in outputs.values())


def test_graphsage_and_rgcn_ablation_encoders_forward() -> None:
    data = _toy_heterodata()
    for encoder in (
        GraphSAGEEncoder(8, 1, data.metadata()),
        HomogeneousGraphSAGEEncoder(8, 1, data.metadata()[0]),
        RGCNEncoder(8, 1, data.metadata()),
    ):
        out = encoder(data.x_dict, data.edge_index_dict)
        assert out["gene"].shape == (8, 8)
        assert out["disease"].shape[1] == 8


def test_hgt_training_loss_decreases_and_evaluates_with_filtered_protocol(
    tmp_path: Path,
) -> None:
    reference, bundle = _split_bundle()
    config = TrainingConfig(
        model_name="hgt",
        hidden_channels=8,
        num_heads=2,
        num_layers=1,
        max_epochs=6,
        patience=6,
        negatives_per_positive=4,
    )
    result = train_one_seed(
        bundle.train_data,
        bundle.val_data,
        bundle.test_data,
        reference,
        seed=13,
        config=config,
        output_dir=tmp_path,
    )
    losses = [row["loss"] for row in result.history]
    metrics = evaluate_split(
        result.model,
        bundle.test_data,
        reference,
        negatives_per_positive=4,
        seed=13,
    )

    assert min(losses) <= losses[0]
    assert {"AUROC", "AUPRC", "filtered_MRR", "filtered_Hits@10"}.issubset(metrics)
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_ablation_models_train_and_multiseed_summary_is_written(tmp_path: Path) -> None:
    reference, bundle = _split_bundle()
    for model_name in ("graphsage", "graphsage_homogeneous", "rgcn"):
        result = train_one_seed(
            bundle.train_data,
            bundle.val_data,
            bundle.test_data,
            reference,
            seed=17,
            config=TrainingConfig(
                model_name=model_name,  # type: ignore[arg-type]
                hidden_channels=8,
                num_layers=1,
                max_epochs=2,
                patience=2,
                negatives_per_positive=4,
            ),
        )
        assert "AUPRC" in result.metrics

    payload = train_multiseed(
        bundle.train_data,
        bundle.val_data,
        bundle.test_data,
        reference,
        config=TrainingConfig(
            model_name="rgcn",
            hidden_channels=8,
            num_layers=1,
            max_epochs=1,
            patience=1,
            negatives_per_positive=4,
        ),
        seeds=(1, 2, 3, 4, 5),
        output_dir=tmp_path,
        hgt_reference={seed: {"AUPRC": 0.1} for seed in (1, 2, 3, 4, 5)},
    )

    assert payload["summary"]["AUPRC"]["mean"] >= 0.0
    assert payload["paired_significance_vs_hgt"]["test"] == "paired_t_test"
    assert (tmp_path / "results_rgcn.json").exists()


def test_model_factory_and_popularity_floor_gate() -> None:
    data = _toy_heterodata()
    model = build_model(
        data.metadata(),
        TrainingConfig(model_name="hgt", hidden_channels=8, num_heads=2, num_layers=1),
    )

    assert isinstance(model, MultiTaskLinkPredictor)
    assert hgt_beats_popularity_floor({"AUPRC": 0.42}, {"AUPRC": 0.2})
