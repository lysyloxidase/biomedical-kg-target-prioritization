from __future__ import annotations

import math

import pytest

from kgtp.hetero.unlabeled_sampling import (
    DegreeMatchedUnlabeledSampler,
    HardUnlabeledSampler,
    RandomUnlabeledSampler,
    SamplingContext,
    TargetProperties,
    mean_property_distance,
)


def _context() -> tuple[SamplingContext, list[tuple[str, str]]]:
    targets = tuple(f"G{index:03d}" for index in range(100))
    properties = {
        target: TargetProperties(
            degree=float(index),
            pagerank=float(index) / 100,
            annotation_count=float(index % 7),
            pathway_count=float(index % 5),
        )
        for index, target in enumerate(targets)
    }
    positives = [("D1", f"G{index:03d}") for index in range(80, 90)]
    held_out = {("D1", "G090"), ("D1", "G091")}
    known = frozenset({*positives, *held_out})
    hard_pool = tuple(f"G{index:03d}" for index in range(60, 80))
    context = SamplingContext(
        source_ids=("D1",),
        target_ids=targets,
        known_positive_pairs=known,
        target_properties=properties,
        hard_candidates_by_source={"D1": hard_pool},
    )
    return context, positives


def test_no_known_or_held_out_positive_is_sampled() -> None:
    context, positives = _context()
    for sampler in (
        RandomUnlabeledSampler(),
        DegreeMatchedUnlabeledSampler(),
        HardUnlabeledSampler(),
    ):
        result = sampler.sample(
            context,
            positives,
            num_samples=10,
            seed=13,
        )
        assert set(result.pairs).isdisjoint(context.known_positive_pairs)


def test_strategies_produce_different_samples_and_distributions() -> None:
    context, positives = _context()
    random_result = RandomUnlabeledSampler().sample(
        context, positives, num_samples=10, seed=13
    )
    degree_result = DegreeMatchedUnlabeledSampler().sample(
        context, positives, num_samples=10, seed=13
    )
    hard_result = HardUnlabeledSampler().sample(
        context, positives, num_samples=10, seed=13
    )

    assert set(random_result.pairs) != set(degree_result.pairs)
    assert set(random_result.pairs) != set(hard_result.pairs)
    assert set(degree_result.pairs) != set(hard_result.pairs)
    assert mean_property_distance(
        positives, degree_result.pairs, context.target_properties
    ) < mean_property_distance(
        positives, random_result.pairs, context.target_properties
    )


def test_same_seed_reproduces_and_different_seed_changes_samples() -> None:
    context, positives = _context()
    for sampler in (
        RandomUnlabeledSampler(),
        DegreeMatchedUnlabeledSampler(),
        HardUnlabeledSampler(),
    ):
        first = sampler.sample(context, positives, num_samples=10, seed=11)
        second = sampler.sample(context, positives, num_samples=10, seed=11)
        third = sampler.sample(context, positives, num_samples=10, seed=12)

        assert first.pairs == second.pairs
        assert first.pairs != third.pairs


def test_degree_matching_is_closer_than_random() -> None:
    context, positives = _context()
    random_pairs = (
        RandomUnlabeledSampler()
        .sample(context, positives, num_samples=10, seed=19)
        .pairs
    )
    matched_pairs = (
        DegreeMatchedUnlabeledSampler()
        .sample(context, positives, num_samples=10, seed=19)
        .pairs
    )

    random_distance = mean_property_distance(
        positives, random_pairs, context.target_properties
    )
    matched_distance = mean_property_distance(
        positives, matched_pairs, context.target_properties
    )
    assert matched_distance < random_distance


def test_hard_candidates_satisfy_declared_pool_rule() -> None:
    context, positives = _context()
    result = HardUnlabeledSampler().sample(context, positives, num_samples=10, seed=5)

    allowed = {("D1", target) for target in context.hard_candidates_by_source["D1"]}
    assert set(result.pairs).issubset(allowed)
    assert result.diagnostics["fallback_count"] == 0


def test_random_sampling_scales_without_cartesian_materialization() -> None:
    sources = tuple(f"D{index}" for index in range(1_000))
    targets = tuple(f"G{index}" for index in range(100_000))
    context = SamplingContext(
        source_ids=sources,
        target_ids=targets,
        known_positive_pairs=frozenset({("D0", "G0")}),
        target_properties={},
        hard_candidates_by_source={},
    )
    result = RandomUnlabeledSampler().sample(
        context,
        [],
        num_samples=25,
        seed=13,
    )

    assert len(result.pairs) == 25
    assert result.diagnostics["cartesian_product_materialized"] is False
    attempts = result.diagnostics["random_draw_attempts"]
    assert isinstance(attempts, int)
    assert attempts < 1_250


def test_sampler_validation_errors_are_explicit() -> None:
    context, positives = _context()
    empty_context = SamplingContext(
        source_ids=(),
        target_ids=(),
        known_positive_pairs=frozenset(),
        target_properties={},
        hard_candidates_by_source={},
    )
    with pytest.raises(ValueError, match="must contain"):
        RandomUnlabeledSampler().sample(
            empty_context,
            [],
            num_samples=1,
            seed=1,
        )
    with pytest.raises(ValueError, match=r"only .* eligible"):
        RandomUnlabeledSampler().sample(
            context,
            positives,
            num_samples=100,
            seed=1,
        )
    with pytest.raises(ValueError, match="requires positive"):
        DegreeMatchedUnlabeledSampler().sample(
            context,
            [],
            num_samples=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="Missing train-only properties"):
        DegreeMatchedUnlabeledSampler().sample(
            context,
            [("D1", "UNKNOWN")],
            num_samples=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="Hard pool has"):
        HardUnlabeledSampler().sample(
            context,
            positives,
            num_samples=100,
            seed=1,
        )


def test_empty_sampling_and_distance_have_defined_results() -> None:
    context, _ = _context()
    result = RandomUnlabeledSampler().sample(
        context,
        [],
        num_samples=0,
        seed=1,
    )

    assert result.pairs == ()
    assert math.isnan(mean_property_distance([], [], context.target_properties))
