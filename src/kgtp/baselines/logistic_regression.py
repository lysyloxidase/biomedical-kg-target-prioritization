"""Logistic regression on node features without message passing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from kgtp.baselines.common import pair_feature_vector, sigmoid
from kgtp.eval.metrics import Triple


class LogisticRegressionBaseline:
    """A tiny NumPy logistic regression baseline for pair features."""

    def __init__(
        self, *, learning_rate: float = 0.2, epochs: int = 200, l2: float = 1e-4
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = np.empty(0, dtype=float)
        self.bias = 0.0
        self.node_features: Mapping[str, Sequence[float]] | None = None

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
        *,
        node_features: Mapping[str, Sequence[float]] | None = None,
    ) -> LogisticRegressionBaseline:
        """Fit on positive and negative pair-feature examples."""

        self.node_features = node_features
        triples = [*positives, *negatives]
        labels = np.asarray(
            [1.0] * len(positives) + [0.0] * len(negatives), dtype=float
        )
        matrix = np.vstack(
            [pair_feature_vector(triple, node_features) for triple in triples]
        )
        self.weights = np.zeros(matrix.shape[1], dtype=float)
        self.bias = 0.0
        for _ in range(self.epochs):
            logits = matrix @ self.weights + self.bias
            predictions = np.asarray(
                [sigmoid(float(value)) for value in logits], dtype=float
            )
            errors = predictions - labels
            grad_w = matrix.T @ errors / len(labels) + self.l2 * self.weights
            grad_b = float(errors.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def score(self, triple: Triple) -> float:
        """Return a logit score for the candidate pair."""

        features = pair_feature_vector(triple, self.node_features)
        if self.weights.size == 0:
            return 0.0
        return float(features @ self.weights + self.bias)
