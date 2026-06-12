from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kgtp.baselines.adjacency_svd import AdjacencySVDBaseline
from kgtp.baselines.common import evaluate_model
from kgtp.baselines.kge import KGE_MODELS, KGEBaseline, score_kge_vectors
from kgtp.baselines.logistic_regression import LogisticRegressionBaseline
from kgtp.baselines.matrix_factorization import MatrixFactorizationBaseline
from kgtp.baselines.node2vec import Node2VecBaseline
from kgtp.baselines.simple import RandomScoreBaseline
from kgtp.baselines.text_embeddings import (
    HashTextBaseline,
    PubMedBERTBaseline,
    SentenceTransformerBaseline,
    load_embedding_cache,
)
from kgtp.eval.metrics import sample_negative_triples


def _baseline_fixture():
    train = [
        ("D1", "associated_with", "G1"),
        ("D1", "associated_with", "G2"),
        ("D2", "associated_with", "G2"),
        ("D2", "associated_with", "G3"),
    ]
    validation = [("D1", "associated_with", "G4")]
    test = [
        ("D1", "associated_with", "G3"),
        ("D2", "associated_with", "G4"),
    ]
    tails = {
        ("D1", "associated_with"): ["G1", "G2", "G3", "G4", "G5", "G6"],
        ("D2", "associated_with"): ["G1", "G2", "G3", "G4", "G5", "G6"],
    }
    all_known = set(train) | set(validation) | set(test)
    train_negatives = sample_negative_triples(
        train,
        all_known=all_known,
        tail_candidates=tails,
        negatives_per_positive=2,
        seed=5,
    )
    validation_negatives = [("D1", "associated_with", "G5")]
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
        validation,
        test,
        all_known,
        tails,
        train_negatives,
        validation_negatives,
        graph_triples,
        descriptions,
        node_features,
    )


def test_supported_baselines_produce_shared_filtered_metrics() -> None:
    (
        train,
        validation,
        test,
        all_known,
        tails,
        train_negatives,
        validation_negatives,
        graph_triples,
        descriptions,
        node_features,
    ) = _baseline_fixture()

    baselines = [
        RandomScoreBaseline(seed=13),
        LogisticRegressionBaseline(epochs=20).fit(
            train, train_negatives, node_features=node_features
        ),
        MatrixFactorizationBaseline(epochs=20, seed=3).fit(train, train_negatives),
        HashTextBaseline().fit(descriptions),
        AdjacencySVDBaseline(dimension=4).fit(graph_triples),
        Node2VecBaseline(
            dimension=4,
            walk_length=5,
            walks_per_node=2,
            epochs=1,
            seed=3,
        ).fit(graph_triples),
        KGEBaseline(model_name="DistMult", epochs=10, seed=3).fit(
            graph_triples,
            train_negatives,
            validation_positives=validation,
            validation_negatives=validation_negatives,
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


def test_adjacency_svd_and_node2vec_are_distinct_algorithms() -> None:
    graph = [
        ("A", "edge", "B"),
        ("B", "edge", "C"),
        ("C", "edge", "D"),
        ("D", "edge", "A"),
    ]
    svd = AdjacencySVDBaseline(dimension=3).fit(graph)
    node2vec = Node2VecBaseline(
        dimension=3,
        walk_length=6,
        walks_per_node=3,
        p=0.5,
        q=2.0,
        epochs=2,
        seed=7,
    ).fit(graph)

    assert node2vec.walks
    assert all(len(walk) <= 6 for walk in node2vec.walks)
    assert node2vec.hyperparameters()["p"] == 0.5
    assert node2vec.hyperparameters()["q"] == 2.0
    assert not torch.allclose(
        torch.tensor(svd.embeddings["A"]),
        torch.tensor(node2vec.embeddings["A"]),
    )


def test_kge_scoring_functions_match_hand_computed_values() -> None:
    head_real = torch.tensor([[1.0, 2.0]])
    relation_real = torch.tensor([[0.5, -1.0]])
    tail_real = torch.tensor([[1.5, 1.0]])
    head_imag = torch.tensor([[0.2, -0.3]])
    relation_imag = torch.tensor([[0.4, 0.1]])
    tail_imag = torch.tensor([[-0.2, 0.5]])

    transe = score_kge_vectors("TransE", head_real, relation_real, tail_real)
    distmult = score_kge_vectors("DistMult", head_real, relation_real, tail_real)
    complex_score = score_kge_vectors(
        "ComplEx",
        head_real,
        relation_real,
        tail_real,
        head_imag=head_imag,
        relation_imag=relation_imag,
        tail_imag=tail_imag,
    )
    rotate = score_kge_vectors(
        "RotatE",
        head_real,
        torch.zeros_like(relation_real),
        tail_real,
        head_imag=head_imag,
        tail_imag=tail_imag,
    )

    assert transe.item() == pytest.approx(0.0)
    assert distmult.item() == pytest.approx(-1.25)
    expected_complex = (
        head_real * relation_real * tail_real
        + head_imag * relation_real * tail_imag
        + head_real * relation_imag * tail_imag
        - head_imag * relation_imag * tail_real
    ).sum()
    assert complex_score.item() == pytest.approx(expected_complex.item())
    expected_rotate = -torch.sqrt(
        (head_real - tail_real).square() + (head_imag - tail_imag).square() + 1e-12
    ).sum()
    assert rotate.item() == pytest.approx(expected_rotate.item())
    assert (
        len({transe.item(), distmult.item(), complex_score.item(), rotate.item()}) == 4
    )


def test_kge_declares_all_four_named_models() -> None:
    assert set(KGE_MODELS) == {"TransE", "DistMult", "ComplEx", "RotatE"}


def test_sentence_transformer_missing_dependency_fails_without_hash_fallback(
    monkeypatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "sentence_transformers":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(
        "kgtp.baselines.text_embeddings.importlib.import_module", fake_import
    )
    with pytest.raises(RuntimeError, match="requires the optional"):
        SentenceTransformerBaseline(model_name="example/model").fit({"D1": "text"})


def test_pubmedbert_missing_dependency_fails_without_hash_fallback(
    monkeypatch,
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "transformers":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(
        "kgtp.baselines.text_embeddings.importlib.import_module", fake_import
    )
    with pytest.raises(RuntimeError, match="requires the optional"):
        PubMedBERTBaseline().fit({"D1": "text"})


def test_sentence_transformer_uses_real_encoder_output_and_cache(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeModel:
        def __init__(self, model_name: str) -> None:
            assert model_name == "test/model"

        def encode(self, texts: list[str]) -> np.ndarray:
            assert texts == ["disease text", "gene text"]
            return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    fake_module = SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setattr(
        "kgtp.baselines.text_embeddings.importlib.import_module",
        lambda name: (
            fake_module
            if name == "sentence_transformers"
            else importlib.import_module(name)
        ),
    )
    cache_path = tmp_path / "embeddings.json"
    model = SentenceTransformerBaseline(
        model_name="test/model",
        cache_path=cache_path,
    ).fit({"G1": "gene text", "D1": "disease text"})

    assert model.score(("D1", "associated_with", "G1")) == pytest.approx(0.0)
    assert model.score(("D1", "associated_with", "missing")) == 0.0
    assert load_embedding_cache(cache_path)["D1"].tolist() == [1.0, 0.0]


def test_pubmedbert_uses_contextual_model_output(monkeypatch) -> None:
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str):
            assert "BiomedBERT" in model_name
            return cls()

        def __call__(self, text: str, **kwargs):
            assert text == "biomedical text"
            assert kwargs["return_tensors"] == "pt"
            return {"attention_mask": torch.tensor([[1, 1]])}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str):
            assert "BiomedBERT" in model_name
            return cls()

        def eval(self) -> None:
            return None

        def __call__(self, **encoded):
            assert "attention_mask" in encoded
            return SimpleNamespace(
                last_hidden_state=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
            )

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModel=FakeModel,
    )
    real_import = importlib.import_module

    def fake_import(name: str):
        if name == "transformers":
            return fake_transformers
        return real_import(name)

    monkeypatch.setattr(
        "kgtp.baselines.text_embeddings.importlib.import_module",
        fake_import,
    )
    model = PubMedBERTBaseline().fit({"D1": "biomedical text"})

    assert model.embeddings["D1"].tolist() == [2.0, 3.0]
    assert model.score(("D1", "associated_with", "D1")) == pytest.approx(1.0)
    assert model.score(("D1", "associated_with", "missing")) == 0.0
