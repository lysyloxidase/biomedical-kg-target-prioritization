from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from kgtp.baselines.centrality import CentralityBaseline
from kgtp.baselines.common import evaluate_model
from kgtp.baselines.kge import KGE_MODELS, KGEBaseline, train_pykeen_pipeline
from kgtp.baselines.logistic_regression import LogisticRegressionBaseline
from kgtp.baselines.matrix_factorization import MatrixFactorizationBaseline
from kgtp.baselines.node2vec import Node2VecBaseline
from kgtp.baselines.popularity import PopularityBaseline
from kgtp.baselines.text_embeddings import TextEmbeddingBaseline
from kgtp.eval.metrics import sample_negative_triples


def _baseline_fixture():
    train = [
        ("D1", "associated_with", "G1"),
        ("D1", "associated_with", "G2"),
        ("D2", "associated_with", "G2"),
        ("D2", "associated_with", "G3"),
    ]
    test = [
        ("D1", "associated_with", "G3"),
        ("D2", "associated_with", "G4"),
    ]
    tails = {
        ("D1", "associated_with"): ["G1", "G2", "G3", "G4", "G5", "G6"],
        ("D2", "associated_with"): ["G1", "G2", "G3", "G4", "G5", "G6"],
    }
    all_known = set(train) | set(test) | {("D1", "associated_with", "G4")}
    train_negatives = sample_negative_triples(
        train,
        all_known=all_known,
        tail_candidates={
            ("D1", "associated_with"): tails[("D1", "associated_with")],
            ("D2", "associated_with"): tails[("D2", "associated_with")],
        },
        negatives_per_positive=2,
        seed=5,
    )
    graph_triples = [
        *train,
        ("G1", "interacts", "G2"),
        ("G2", "interacts", "G3"),
        ("G3", "participates_in", "P1"),
        ("G4", "participates_in", "P1"),
    ]
    descriptions = {
        "D1": "knee osteoarthritis cartilage pain",
        "D2": "osteoarthritis inflammation",
        "G1": "cartilage matrix anabolic gene",
        "G2": "matrix remodeling gene",
        "G3": "inflammatory cartilage target",
        "G4": "pain and joint biology target",
        "G5": "background gene",
        "G6": "background protein",
    }
    node_features = {
        node_id: [float(index), float(index % 2), 1.0]
        for index, node_id in enumerate(descriptions)
    }
    return (
        train,
        test,
        all_known,
        tails,
        train_negatives,
        graph_triples,
        descriptions,
        node_features,
    )


def test_all_seven_baselines_produce_filtered_protocol_metrics(tmp_path) -> None:
    (
        train,
        test,
        all_known,
        tails,
        train_negatives,
        graph_triples,
        descriptions,
        node_features,
    ) = _baseline_fixture()

    baselines = [
        PopularityBaseline().fit(graph_triples),
        LogisticRegressionBaseline(epochs=20).fit(
            train, train_negatives, node_features=node_features
        ),
        MatrixFactorizationBaseline(epochs=20, seed=3).fit(train, train_negatives),
        TextEmbeddingBaseline(cache_path=tmp_path / "text_cache.json").fit(
            descriptions
        ),
        Node2VecBaseline(dimension=4).fit(graph_triples),
        CentralityBaseline().fit(graph_triples),
        KGEBaseline(model_name="DistMult", epochs=20, seed=3).fit(
            train, train_negatives
        ),
    ]

    for model in baselines:
        result = evaluate_model(
            model,
            test,
            all_known=all_known,
            tail_candidates=tails,
            negatives_per_positive=1_000,
            seed=13,
        )
        assert result["primary_metric"] == "AUPRC"
        assert {"AUROC", "AUPRC", "filtered", "raw"}.issubset(result)
        assert {"Hits@1", "Hits@3", "Hits@10", "MRR"}.issubset(result["filtered"])  # type: ignore[arg-type]

    assert (tmp_path / "text_cache.json").exists()


def test_kge_declares_ogbl_biokg_competitor_models() -> None:
    assert set(KGE_MODELS) == {"TransE", "DistMult", "ComplEx", "RotatE"}


def test_pykeen_pipeline_adapter_is_called_when_available(monkeypatch) -> None:
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metric_results={"mrr": 0.5})

    def fake_import(name: str):
        if name == "pykeen.pipeline":
            return SimpleNamespace(pipeline=fake_pipeline)
        return importlib.import_module(name)

    monkeypatch.setattr("kgtp.baselines.kge.importlib.import_module", fake_import)

    result = train_pykeen_pipeline(
        [("D1", "associated_with", "G1")],
        model_name="TransE",
        seed=13,
    )

    assert result.metric_results["mrr"] == 0.5
    assert captured["model"] == "TransE"
    assert captured["random_seed"] == 13


def test_pykeen_adapter_reports_missing_dependency(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "pykeen.pipeline":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr("kgtp.baselines.kge.importlib.import_module", fake_import)

    with pytest.raises(RuntimeError, match="PyKEEN is not installed"):
        train_pykeen_pipeline(
            [("D1", "associated_with", "G1")], model_name="TransE", seed=13
        )
