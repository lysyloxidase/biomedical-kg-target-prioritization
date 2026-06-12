"""Simple honest baseline scorers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from kgtp.eval.metrics import Triple


class RandomScoreBaseline:
    """Deterministic pseudorandom scorer used as a chance-level control."""

    def __init__(self, *, seed: int = 13) -> None:
        self.seed = seed

    def score(self, triple: Triple) -> float:
        """Hash the triple and seed to a stable score in [0, 1)."""

        payload = f"{self.seed}\0{triple[0]}\0{triple[1]}\0{triple[2]}"
        digest = hashlib.sha256(payload.encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64


class SourceScoreBaseline:
    """Use source-provided scores only for train-observed target edges."""

    def __init__(self) -> None:
        self.scores: dict[Triple, float] = {}

    def fit(
        self,
        train_scores: Mapping[Triple, float],
    ) -> SourceScoreBaseline:
        """Store scores for train-known edges without consulting held-out scores."""

        self.scores = {triple: float(score) for triple, score in train_scores.items()}
        return self

    def score(self, triple: Triple) -> float:
        """Return zero for every edge not observed in the train partitions."""

        return self.scores.get(triple, 0.0)


class TargetPopularityBaseline:
    """Score genes by train-graph degree supplied explicitly by the caller."""

    def __init__(self) -> None:
        self.target_scores: dict[str, float] = {}

    def fit(
        self,
        target_scores: Mapping[str, float],
    ) -> TargetPopularityBaseline:
        """Store train-only target popularity values."""

        self.target_scores = {
            target: float(score) for target, score in target_scores.items()
        }
        return self

    def score(self, triple: Triple) -> float:
        """Return train-only popularity for the candidate target."""

        return self.target_scores.get(triple[2], 0.0)


def triples_from_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    relation: str = "associated_with",
) -> list[Triple]:
    """Convert source-target pairs to typed evaluation triples."""

    return [(source, relation, target) for source, target in pairs]
