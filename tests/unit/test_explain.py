from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation

from kgtp.explain.attention import extract_hgt_attention_weights
from kgtp.explain.case_studies import (
    PredictionCandidate,
    build_phase6_case_studies,
    select_known_target_index,
    select_novel_prediction,
)
from kgtp.explain.explainer import DISEASE_GENE_EDGE, TargetExplainer
from kgtp.explain.metapaths import rank_metapaths
from kgtp.models.hgt import HGTEncoder
from kgtp.models.multitask import MultiTaskLinkPredictor


class _LinearGeneLinkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.is_trained = False

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del edge_index_dict
        return x_dict

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        del edge_type
        genes = z_dict["gene"][edge_label_index[1]]
        return genes @ self.weight + self.bias


def _toy_explain_data() -> HeteroData:
    data = HeteroData()
    data["disease"].x = torch.eye(1)
    data["disease"].node_id = ["EFO_0004616"]
    data["disease"].label = ["knee osteoarthritis"]
    data["gene"].x = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.1],
            [0.8, 0.1, 0.3, 0.1],
            [0.2, 1.0, 0.4, 0.7],
            [0.1, 0.9, 0.6, 0.2],
            [0.2, 0.2, 1.0, 0.4],
        ],
        dtype=torch.float32,
    )
    data["gene"].node_id = [
        "ENSG_GDF5",
        "ENSG_MMP13",
        "ENSG_NOVEL1",
        "ENSG_WNT5A",
        "ENSG_COL2A1",
    ]
    data["gene"].symbol = ["GDF5", "MMP13", "NOVEL1", "WNT5A", "COL2A1"]
    data["pathway"].x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
    )
    data["pathway"].node_id = ["R-HSA-WNT", "R-HSA-BMP", "R-HSA-ECM"]
    data["pathway"].label = [
        "Wnt beta-catenin signaling",
        "TGF-beta BMP cartilage differentiation",
        "extracellular matrix collagen remodeling",
    ]

    disease_gene = torch.tensor([[0, 0, 0], [0, 1, 3]], dtype=torch.long)
    gene_pathway = torch.tensor(
        [[0, 2, 3, 2, 1, 4], [1, 1, 0, 0, 2, 2]],
        dtype=torch.long,
    )
    ppi = torch.tensor([[0, 3, 1], [2, 2, 4]], dtype=torch.long)

    data[DISEASE_GENE_EDGE].edge_index = disease_gene
    data[("gene", "rev_associated_with", "disease")].edge_index = disease_gene.flip(0)
    data[("gene", "participates_in", "pathway")].edge_index = gene_pathway
    data[("pathway", "rev_participates_in", "gene")].edge_index = gene_pathway.flip(0)
    data[("gene", "interacts", "gene")].edge_index = ppi
    return data


def _toy_model(data: HeteroData) -> MultiTaskLinkPredictor:
    torch.manual_seed(7)
    model = MultiTaskLinkPredictor(
        HGTEncoder(8, 2, 1, data.metadata()),
        hidden_channels=8,
        edge_types=(DISEASE_GENE_EDGE,),
    )
    model.is_trained = True
    return model


def test_captum_target_explainer_returns_hetero_explanation() -> None:
    data = _toy_explain_data()
    explainer = TargetExplainer(
        _toy_model(data),
        data,
        integration_steps=2,
        use_pyg_captum=False,
    )

    explanation = explainer.explain_link(0, 2)
    subgraph = explainer.explanatory_subgraph(explanation)

    assert isinstance(explanation, HeteroExplanation)
    assert explanation.algorithm == "CaptumExplainer"
    assert explanation["gene"].node_mask.shape[0] == 5
    assert explanation["gene"].feature_mask.shape[0] == 4
    assert subgraph["prediction"] == {
        "edge_type": list(DISEASE_GENE_EDGE),
        "disease_idx": 0,
        "gene_idx": 2,
    }
    assert any(node["node_id"] == "ENSG_NOVEL1" for node in subgraph["nodes"])  # type: ignore[index]


def test_attention_explainer_extracts_hgt_attention_per_prediction() -> None:
    data = _toy_explain_data()
    attention = extract_hgt_attention_weights(_toy_model(data), data, 0, 2, top_k=6)

    assert attention["algorithm"] == "endpoint_conditioned_topology_proxy"
    assert attention["model_attribution"] is False
    assert attention["attention_available"] is False
    assert attention["weights"]
    summary = cast(Mapping[str, float], attention["meta_relation_summary"])
    weights = cast(list[Mapping[str, object]], attention["weights"])
    assert "disease__associated_with__gene" in summary
    assert all("layer" in row for row in weights)


def test_metapath_explanation_surfaces_pathway_and_ppi_paths() -> None:
    data = _toy_explain_data()
    paths = rank_metapaths(data, 0, 2, max_paths=10)
    schemas = {path.schema for path in paths}

    assert "disease->gene->pathway->gene" in schemas
    assert "disease->gene->PPI->gene" in schemas
    assert any("BMP" in node.label for path in paths for node in path.nodes)
    assert all(path.score > 0 for path in paths)


def test_case_studies_write_known_and_novel_hypothesis_figures(tmp_path: Path) -> None:
    data = _toy_explain_data()
    explainer = TargetExplainer(
        _toy_model(data),
        data,
        integration_steps=1,
        use_pyg_captum=False,
    )
    predictions = [
        PredictionCandidate(0, 2, 0.99),
        PredictionCandidate(0, 4, 0.75),
        PredictionCandidate(0, 3, 0.5),
    ]

    assert select_known_target_index(data) == 0
    assert (
        select_novel_prediction(
            predictions,
            {(0, 0), (0, 1), (0, 3)},
            disease_idx=0,
            num_genes=5,
        ).gene_idx
        == 2
    )

    results = build_phase6_case_studies(
        explainer,
        data,
        predictions,
        {(0, 0), (0, 1), (0, 3)},
        output_dir=tmp_path,
    )

    assert len(results) >= 3
    assert (tmp_path / "case_studies.json").exists()
    assert (tmp_path / "case_study_narrative.md").exists()
    narrative = (tmp_path / "case_study_narrative.md").read_text(encoding="utf-8")
    assert "computational hypothesis" in narrative
    assert "TGF-beta" in narrative or "wnt" in narrative.lower()
    assert any(
        result.is_known_target and result.gene_symbol == "GDF5" for result in results
    )
    assert any(
        result.is_hypothesis and result.gene_symbol == "NOVEL1" for result in results
    )
    for result in results:
        assert Path(result.figure_paths["png"]).exists()
        assert Path(result.figure_paths["pdf"]).exists()


def test_untrained_model_cannot_produce_explanations() -> None:
    data = _toy_explain_data()
    model = _toy_model(data)
    model.is_trained = False

    explainer = TargetExplainer(model, data, use_pyg_captum=False)

    with pytest.raises(RuntimeError, match="trained model"):
        explainer.explain_link(0, 2)


def test_parameter_randomization_changes_model_attributions() -> None:
    data = _toy_explain_data()
    model = _toy_model(data)
    explainer = TargetExplainer(
        model,
        data,
        integration_steps=2,
        use_pyg_captum=False,
    )
    before = explainer.explain_link(0, 2)["gene"].feature_mask.clone()

    torch.manual_seed(91)
    for parameter in model.parameters():
        parameter.data.normal_()
    after = explainer.explain_link(0, 2)["gene"].feature_mask

    assert not torch.allclose(before, after)


def test_removing_highly_attributed_edge_changes_prediction_score() -> None:
    data = _toy_explain_data()
    explainer = TargetExplainer(
        _toy_model(data),
        data,
        integration_steps=1,
        use_pyg_captum=False,
    )
    explanation = explainer.explain_link(0, 2)
    best: tuple[tuple[str, str, str], int, float] | None = None
    for edge_type in explanation.edge_types:
        mask = explanation[edge_type].edge_mask
        if mask.numel() == 0:
            continue
        position = int(mask.argmax().item())
        value = float(mask[position].item())
        if best is None or value > best[2]:
            best = (cast(tuple[str, str, str], edge_type), position, value)

    assert best is not None
    assert best[2] > 0
    baseline = explainer.score_link(0, 2)
    perturbed = explainer.score_without_edges(0, 2, {best[0]: [best[1]]})
    assert perturbed != pytest.approx(baseline)


def test_held_out_candidate_edge_is_absent_from_explanation_message_graph() -> None:
    data = _toy_explain_data()
    explainer = TargetExplainer(
        _toy_model(data),
        data,
        integration_steps=1,
        use_pyg_captum=False,
    )

    explanation = explainer.explain_link(0, 2)
    message_pairs = {
        tuple(pair) for pair in explanation[DISEASE_GENE_EDGE].edge_index.t().tolist()
    }
    reverse_pairs = {
        tuple(pair)
        for pair in explanation[("gene", "rev_associated_with", "disease")]
        .edge_index.t()
        .tolist()
    }

    assert (0, 2) not in message_pairs
    assert (2, 0) not in reverse_pairs


def test_label_randomization_reduces_controlled_attribution_signal() -> None:
    data = HeteroData()
    data["disease"].x = torch.ones(1, 1)
    data["disease"].node_id = ["EFO_CONTROL"]
    data["gene"].x = torch.tensor(
        [
            [2.0, 0.2],
            [1.6, 0.3],
            [0.3, 1.6],
            [0.2, 2.0],
            [1.2, 0.8],
            [0.8, 1.2],
        ]
    )
    data["gene"].node_id = [f"GENE_{index}" for index in range(6)]
    data[DISEASE_GENE_EDGE].edge_index = torch.empty((2, 0), dtype=torch.long)
    clean_labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    randomized_labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    clean_model = _fit_linear_control(data["gene"].x, clean_labels)
    randomized_model = _fit_linear_control(data["gene"].x, randomized_labels)
    clean_explanation = TargetExplainer(
        clean_model,
        data,
        integration_steps=8,
        use_pyg_captum=False,
    ).explain_link(0, 0)
    randomized_explanation = TargetExplainer(
        randomized_model,
        data,
        integration_steps=8,
        use_pyg_captum=False,
    ).explain_link(0, 0)

    clean_signal = clean_explanation["gene"].raw_feature_attribution.abs().sum()
    randomized_signal = (
        randomized_explanation["gene"].raw_feature_attribution.abs().sum()
    )
    assert clean_signal > randomized_signal * 20


def _fit_linear_control(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> _LinearGeneLinkModel:
    torch.manual_seed(3)
    model = _LinearGeneLinkModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        logits = features @ model.weight + model.bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    model.is_trained = True
    return model
