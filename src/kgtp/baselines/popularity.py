"""Popularity / degree baseline for the must-beat floor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from kgtp.eval.metrics import Triple


class PopularityBaseline:
    """Score a candidate by global target popularity and endpoint degree."""

    def __init__(self) -> None:
        self.tail_counts: Counter[str] = Counter()
        self.degree: Counter[str] = Counter()

    def fit(self, triples: Sequence[Triple]) -> PopularityBaseline:
        """Fit frequency counts from training triples."""

        self.tail_counts.clear()
        self.degree.clear()
        for head, _, tail in triples:
            self.tail_counts[tail] += 1
            self.degree[head] += 1
            self.degree[tail] += 1
        return self

    def score(self, triple: Triple) -> float:
        """Score by target association frequency and graph degree."""

        _, _, tail = triple
        return float(self.tail_counts[tail] + self.degree[tail])
