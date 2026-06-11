"""Classical matrix-factorization baseline for disease-gene links."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kgtp.baselines.common import sigmoid
from kgtp.eval.metrics import Triple


class MatrixFactorizationBaseline:
    """Logistic matrix factorization over source-tail association pairs."""

    def __init__(
        self,
        *,
        rank: int = 8,
        learning_rate: float = 0.1,
        epochs: int = 200,
        seed: int = 13,
    ) -> None:
        self.rank = rank
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.source_index: dict[str, int] = {}
        self.tail_index: dict[str, int] = {}
        self.source_factors = np.empty((0, rank), dtype=float)
        self.tail_factors = np.empty((0, rank), dtype=float)

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
    ) -> MatrixFactorizationBaseline:
        """Fit latent factors with pointwise logistic loss."""

        sources = sorted({head for head, _, _ in [*positives, *negatives]})
        tails = sorted({tail for _, _, tail in [*positives, *negatives]})
        self.source_index = {source: index for index, source in enumerate(sources)}
        self.tail_index = {tail: index for index, tail in enumerate(tails)}
        rng = np.random.default_rng(self.seed)
        self.source_factors = rng.normal(0.0, 0.1, size=(len(sources), self.rank))
        self.tail_factors = rng.normal(0.0, 0.1, size=(len(tails), self.rank))
        examples = [
            *[(triple, 1.0) for triple in positives],
            *[(triple, 0.0) for triple in negatives],
        ]
        for _ in range(self.epochs):
            rng.shuffle(examples)
            for (head, _, tail), label in examples:
                source_id = self.source_index[head]
                tail_id = self.tail_index[tail]
                source_vec = self.source_factors[source_id].copy()
                tail_vec = self.tail_factors[tail_id].copy()
                error = sigmoid(float(source_vec @ tail_vec)) - label
                self.source_factors[source_id] -= self.learning_rate * error * tail_vec
                self.tail_factors[tail_id] -= self.learning_rate * error * source_vec
        return self

    def score(self, triple: Triple) -> float:
        """Return dot-product latent compatibility."""

        head, _, tail = triple
        if head not in self.source_index or tail not in self.tail_index:
            return 0.0
        return float(
            self.source_factors[self.source_index[head]]
            @ self.tail_factors[self.tail_index[tail]]
        )
