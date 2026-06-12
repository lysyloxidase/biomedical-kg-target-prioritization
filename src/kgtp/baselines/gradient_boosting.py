"""Small gradient-boosted regression trees for pair classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from kgtp.baselines.common import pair_feature_vector, sigmoid
from kgtp.eval.metrics import Triple


@dataclass(frozen=True)
class DecisionStump:
    """One-dimensional regression stump."""

    feature: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Return stump outputs for a feature matrix."""

        return np.where(
            matrix[:, self.feature] <= self.threshold,
            self.left_value,
            self.right_value,
        )


class GradientBoostedTreesBaseline:
    """Logistic gradient boosting with deterministic decision stumps."""

    def __init__(
        self,
        *,
        estimators: int = 30,
        learning_rate: float = 0.1,
    ) -> None:
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.bias = 0.0
        self.stumps: list[DecisionStump] = []
        self.node_features: Mapping[str, Sequence[float]] | None = None

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
        *,
        node_features: Mapping[str, Sequence[float]],
    ) -> GradientBoostedTreesBaseline:
        """Fit stumps to logistic residuals using explicit train-fitted features."""

        self.node_features = node_features
        triples = [*positives, *negatives]
        labels = np.asarray(
            [1.0] * len(positives) + [0.0] * len(negatives), dtype=float
        )
        matrix = np.vstack(
            [pair_feature_vector(triple, node_features) for triple in triples]
        )
        prevalence = min(max(float(labels.mean()), 1e-6), 1.0 - 1e-6)
        self.bias = float(np.log(prevalence / (1.0 - prevalence)))
        logits = np.full(len(labels), self.bias, dtype=float)
        self.stumps = []
        for _ in range(self.estimators):
            probabilities = np.asarray([sigmoid(value) for value in logits])
            residuals = labels - probabilities
            stump = _fit_stump(matrix, residuals)
            self.stumps.append(stump)
            logits += self.learning_rate * stump.predict(matrix)
        return self

    def score(self, triple: Triple) -> float:
        """Return boosted classification logit."""

        if self.node_features is None:
            msg = "Gradient-boosted baseline must be fitted before scoring"
            raise RuntimeError(msg)
        row = pair_feature_vector(triple, self.node_features).reshape(1, -1)
        return float(
            self.bias
            + sum(
                self.learning_rate * float(stump.predict(row)[0])
                for stump in self.stumps
            )
        )


def _fit_stump(matrix: np.ndarray, residuals: np.ndarray) -> DecisionStump:
    best: tuple[float, DecisionStump] | None = None
    for feature in range(matrix.shape[1]):
        values = np.unique(matrix[:, feature])
        thresholds = values if len(values) == 1 else (values[:-1] + values[1:]) / 2
        for threshold in thresholds:
            left = matrix[:, feature] <= threshold
            right = ~left
            left_value = float(residuals[left].mean()) if left.any() else 0.0
            right_value = float(residuals[right].mean()) if right.any() else 0.0
            predictions = np.where(left, left_value, right_value)
            error = float(np.square(residuals - predictions).sum())
            stump = DecisionStump(
                feature=feature,
                threshold=float(threshold),
                left_value=left_value,
                right_value=right_value,
            )
            if best is None or error < best[0]:
                best = (error, stump)
    if best is None:
        msg = "Cannot fit a decision stump to an empty feature matrix"
        raise ValueError(msg)
    return best[1]
