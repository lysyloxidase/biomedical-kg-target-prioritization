"""Train-only unlabeled-pair sampling for biomedical link prediction."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Pair = tuple[str, str]


@dataclass(frozen=True)
class TargetProperties:
    """Train-only properties used by degree-matched sampling."""

    degree: float
    pagerank: float
    annotation_count: float
    pathway_count: float

    def vector(self) -> tuple[float, float, float, float]:
        """Return a stable numeric representation."""

        return (
            self.degree,
            self.pagerank,
            self.annotation_count,
            self.pathway_count,
        )


@dataclass(frozen=True)
class SamplingContext:
    """Candidate universe and train-only sampling state."""

    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    known_positive_pairs: frozenset[Pair]
    target_properties: Mapping[str, TargetProperties]
    hard_candidates_by_source: Mapping[str, tuple[str, ...]]
    hard_rule: str = "same_pathway_or_ppi_neighbor_or_shared_go"


@dataclass(frozen=True)
class SamplingResult:
    """Sampled unlabeled pairs with auditable diagnostics."""

    strategy: str
    pairs: tuple[Pair, ...]
    diagnostics: dict[str, object]


class RandomUnlabeledSampler:
    """Rejection-sample unobserved pairs without a Cartesian-product table."""

    strategy = "random_unlabeled"

    def sample(
        self,
        context: SamplingContext,
        positives: Sequence[Pair],
        *,
        num_samples: int,
        seed: int,
        excluded_pairs: set[Pair] | None = None,
    ) -> SamplingResult:
        """Sample deterministic unobserved pairs from the configured universe."""

        del positives
        pairs, attempts = _sample_random_pairs(
            context,
            num_samples=num_samples,
            seed=seed,
            excluded_pairs=excluded_pairs,
        )
        return SamplingResult(
            strategy=self.strategy,
            pairs=tuple(sorted(pairs)),
            diagnostics={
                "strategy": self.strategy,
                "seed": seed,
                "requested": num_samples,
                "sampled": len(pairs),
                "random_draw_attempts": attempts,
                "cartesian_product_materialized": False,
                "label_semantics": "unlabeled_not_confirmed_negative",
            },
        )


class DegreeMatchedUnlabeledSampler:
    """Match each positive target to train-only topology/annotation properties."""

    strategy = "degree_matched_unlabeled"

    def sample(
        self,
        context: SamplingContext,
        positives: Sequence[Pair],
        *,
        num_samples: int,
        seed: int,
        excluded_pairs: set[Pair] | None = None,
    ) -> SamplingResult:
        """Choose candidates minimizing standardized distance to positive targets."""

        if not positives:
            msg = "Degree-matched sampling requires positive reference pairs"
            raise ValueError(msg)
        forbidden = set(context.known_positive_pairs)
        forbidden.update(excluded_pairs or set())
        scales = _property_scales(context.target_properties)
        rng = random.Random(seed)
        selected: list[Pair] = []
        distances: list[float] = []
        ordered_positives = list(positives)
        rng.shuffle(ordered_positives)
        cursor = 0
        while len(selected) < num_samples:
            source, positive_target = ordered_positives[cursor % len(ordered_positives)]
            cursor += 1
            positive_properties = context.target_properties.get(positive_target)
            if positive_properties is None:
                msg = f"Missing train-only properties for positive target {positive_target}"
                raise ValueError(msg)
            candidates: list[tuple[float, str]] = []
            for target in context.target_ids:
                pair = (source, target)
                properties = context.target_properties.get(target)
                if pair in forbidden or properties is None:
                    continue
                candidates.append(
                    (
                        _property_distance(
                            positive_properties,
                            properties,
                            scales,
                        ),
                        target,
                    )
                )
            if not candidates:
                msg = "Degree-matched sampler exhausted eligible candidates"
                raise ValueError(msg)
            candidates.sort(key=lambda item: (item[0], item[1]))
            nearest_distance = candidates[0][0]
            nearest = [
                item
                for item in candidates
                if math.isclose(item[0], nearest_distance, rel_tol=1e-12, abs_tol=1e-12)
            ]
            distance, target = rng.choice(nearest)
            pair = (source, target)
            forbidden.add(pair)
            selected.append(pair)
            distances.append(distance)
        return SamplingResult(
            strategy=self.strategy,
            pairs=tuple(sorted(selected)),
            diagnostics={
                "strategy": self.strategy,
                "seed": seed,
                "requested": num_samples,
                "sampled": len(selected),
                "mean_standardized_property_distance": sum(distances) / len(distances),
                "matched_properties": [
                    "degree",
                    "pagerank",
                    "annotation_count",
                    "pathway_count",
                ],
                "cartesian_product_materialized": False,
                "label_semantics": "unlabeled_not_confirmed_negative",
            },
        )


class HardUnlabeledSampler:
    """Sample candidates satisfying an explicit train-only biological rule."""

    strategy = "hard_unlabeled"

    def sample(
        self,
        context: SamplingContext,
        positives: Sequence[Pair],
        *,
        num_samples: int,
        seed: int,
        excluded_pairs: set[Pair] | None = None,
    ) -> SamplingResult:
        """Draw only from source-specific train-derived hard pools."""

        del positives
        forbidden = set(context.known_positive_pairs)
        forbidden.update(excluded_pairs or set())
        candidates = [
            (source, target)
            for source in context.source_ids
            for target in context.hard_candidates_by_source.get(source, ())
            if (source, target) not in forbidden
        ]
        candidates = sorted(set(candidates))
        if len(candidates) < num_samples:
            msg = (
                f"Hard pool has {len(candidates)} eligible pairs; "
                f"{num_samples} requested"
            )
            raise ValueError(msg)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        selected = sorted(candidates[:num_samples])
        return SamplingResult(
            strategy=self.strategy,
            pairs=tuple(selected),
            diagnostics={
                "strategy": self.strategy,
                "seed": seed,
                "requested": num_samples,
                "sampled": len(selected),
                "hard_rule": context.hard_rule,
                "eligible_hard_pool_size": len(candidates),
                "fallback_count": 0,
                "cartesian_product_materialized": False,
                "label_semantics": "unlabeled_not_confirmed_negative",
            },
        )


def mean_property_distance(
    positives: Sequence[Pair],
    sampled: Sequence[Pair],
    properties: Mapping[str, TargetProperties],
) -> float:
    """Compare target-property means between positive and sampled pairs."""

    if not positives or not sampled:
        return math.nan
    positive_mean = _mean_vector(
        [properties[target].vector() for _, target in positives]
    )
    sampled_mean = _mean_vector([properties[target].vector() for _, target in sampled])
    scales = _property_scales(properties)
    return math.sqrt(
        sum(
            ((positive - candidate) / scale) ** 2
            for positive, candidate, scale in zip(
                positive_mean, sampled_mean, scales, strict=True
            )
        )
    )


def _sample_random_pairs(
    context: SamplingContext,
    *,
    num_samples: int,
    seed: int,
    excluded_pairs: set[Pair] | None,
) -> tuple[list[Pair], int]:
    if num_samples <= 0:
        return [], 0
    if not context.source_ids or not context.target_ids:
        msg = "Sampling universe must contain source and target IDs"
        raise ValueError(msg)
    forbidden = set(context.known_positive_pairs)
    forbidden.update(excluded_pairs or set())
    total_pairs = len(context.source_ids) * len(context.target_ids)
    source_set = set(context.source_ids)
    target_set = set(context.target_ids)
    eligible_count = total_pairs - sum(
        source in source_set and target in target_set for source, target in forbidden
    )
    if eligible_count < num_samples:
        msg = f"Requested {num_samples} pairs but only {eligible_count} are eligible"
        raise ValueError(msg)

    rng = random.Random(seed)
    selected: set[Pair] = set()
    attempts = 0
    max_random_attempts = max(100, num_samples * 50)
    while len(selected) < num_samples and attempts < max_random_attempts:
        pair = (
            context.source_ids[rng.randrange(len(context.source_ids))],
            context.target_ids[rng.randrange(len(context.target_ids))],
        )
        attempts += 1
        if pair not in forbidden:
            selected.add(pair)

    if len(selected) < num_samples:
        start = rng.randrange(total_pairs)
        step = _coprime_step(total_pairs, rng)
        for offset in range(total_pairs):
            flat_index = (start + offset * step) % total_pairs
            source = context.source_ids[flat_index // len(context.target_ids)]
            target = context.target_ids[flat_index % len(context.target_ids)]
            pair = (source, target)
            if pair not in forbidden:
                selected.add(pair)
                if len(selected) == num_samples:
                    break
    return sorted(selected), attempts


def _coprime_step(total: int, rng: random.Random) -> int:
    if total <= 1:
        return 1
    candidate = rng.randrange(1, total)
    while math.gcd(candidate, total) != 1:
        candidate = (candidate + 1) % total or 1
    return candidate


def _property_scales(
    properties: Mapping[str, TargetProperties],
) -> tuple[float, float, float, float]:
    vectors = [value.vector() for value in properties.values()]
    if not vectors:
        return (1.0, 1.0, 1.0, 1.0)
    means = _mean_vector(vectors)
    variances = [
        sum((vector[index] - means[index]) ** 2 for vector in vectors) / len(vectors)
        for index in range(4)
    ]
    return tuple(max(math.sqrt(variance), 1e-12) for variance in variances)  # type: ignore[return-value]


def _property_distance(
    first: TargetProperties,
    second: TargetProperties,
    scales: tuple[float, float, float, float],
) -> float:
    return math.sqrt(
        sum(
            ((left - right) / scale) ** 2
            for left, right, scale in zip(
                first.vector(), second.vector(), scales, strict=True
            )
        )
    )


def _mean_vector(
    vectors: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(4)
    )  # type: ignore[return-value]
