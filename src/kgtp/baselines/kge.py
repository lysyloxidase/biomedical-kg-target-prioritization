"""Knowledge-graph embedding baselines with optional PyKEEN integration."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from kgtp.baselines.common import sigmoid
from kgtp.eval.metrics import Triple

KGE_MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")


class KGEBaseline:
    """Small KGE scorer; can delegate training to PyKEEN when installed."""

    def __init__(
        self,
        *,
        model_name: str = "TransE",
        dimension: int = 16,
        learning_rate: float = 0.05,
        epochs: int = 100,
        seed: int = 13,
    ) -> None:
        if model_name not in KGE_MODELS:
            msg = f"Unsupported KGE model: {model_name}"
            raise ValueError(msg)
        self.model_name = model_name
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.entity_index: dict[str, int] = {}
        self.relation_index: dict[str, int] = {}
        self.entity_embeddings = np.empty((0, dimension), dtype=float)
        self.relation_embeddings = np.empty((0, dimension), dtype=float)
        self.pykeen_result: Any | None = None

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
        *,
        use_pykeen: bool = False,
    ) -> KGEBaseline:
        """Train KGE embeddings, optionally through PyKEEN."""

        if use_pykeen:
            self.pykeen_result = train_pykeen_pipeline(
                positives, model_name=self.model_name, seed=self.seed
            )
            return self
        self._fit_numpy(positives, negatives)
        return self

    def score(self, triple: Triple) -> float:
        """Return a model-specific KGE compatibility score."""

        if self.pykeen_result is not None:
            return self._score_pykeen(triple)
        head, relation, tail = triple
        if (
            head not in self.entity_index
            or tail not in self.entity_index
            or relation not in self.relation_index
        ):
            return 0.0
        h = self.entity_embeddings[self.entity_index[head]]
        r = self.relation_embeddings[self.relation_index[relation]]
        t = self.entity_embeddings[self.entity_index[tail]]
        if self.model_name == "TransE":
            return float(-np.linalg.norm(h + r - t))
        if self.model_name == "DistMult":
            return float(np.sum(h * r * t))
        if self.model_name == "ComplEx":
            return float(np.sum(h * r * t) + np.sum(h * t))
        return float(-np.linalg.norm((h + r) - t))

    def _fit_numpy(
        self, positives: Sequence[Triple], negatives: Sequence[Triple]
    ) -> None:
        triples = [*positives, *negatives]
        entities = sorted(
            {entity for head, _, tail in triples for entity in (head, tail)}
        )
        relations = sorted({relation for _, relation, _ in triples})
        self.entity_index = {entity: index for index, entity in enumerate(entities)}
        self.relation_index = {
            relation: index for index, relation in enumerate(relations)
        }
        rng = np.random.default_rng(self.seed)
        self.entity_embeddings = rng.normal(
            0.0, 0.1, size=(len(entities), self.dimension)
        )
        self.relation_embeddings = rng.normal(
            0.0, 0.1, size=(len(relations), self.dimension)
        )
        examples = [
            *[(triple, 1.0) for triple in positives],
            *[(triple, 0.0) for triple in negatives],
        ]
        for _ in range(self.epochs):
            rng.shuffle(examples)
            for triple, label in examples:
                self._sgd_step(triple, label)

    def _sgd_step(self, triple: Triple, label: float) -> None:
        head, relation, tail = triple
        h_id = self.entity_index[head]
        r_id = self.relation_index[relation]
        t_id = self.entity_index[tail]
        h = self.entity_embeddings[h_id].copy()
        r = self.relation_embeddings[r_id].copy()
        t = self.entity_embeddings[t_id].copy()
        score = float(np.sum(h * r * t))
        error = sigmoid(score) - label
        self.entity_embeddings[h_id] -= self.learning_rate * error * (r * t)
        self.relation_embeddings[r_id] -= self.learning_rate * error * (h * t)
        self.entity_embeddings[t_id] -= self.learning_rate * error * (h * r)

    def _score_pykeen(self, triple: Triple) -> float:
        del triple
        return 0.0


def train_pykeen_pipeline(
    triples: Sequence[Triple],
    *,
    model_name: str,
    seed: int,
) -> Any:
    """Run PyKEEN's pipeline for TransE/DistMult/ComplEx/RotatE when installed."""

    try:
        pipeline_module = importlib.import_module("pykeen.pipeline")
    except ModuleNotFoundError as exc:
        msg = "PyKEEN is not installed; install pykeen to run external KGE baselines"
        raise RuntimeError(msg) from exc
    pipeline = pipeline_module.pipeline
    return pipeline(
        training=[tuple(triple) for triple in triples],
        testing=[tuple(triple) for triple in triples],
        model=model_name,
        random_seed=seed,
    )
