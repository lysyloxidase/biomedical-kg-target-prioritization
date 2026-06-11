"""OGB-style filtered link-prediction metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias, cast

import numpy as np

Triple: TypeAlias = tuple[str, str, str]
Query: TypeAlias = tuple[str, str]
ScoreMap: TypeAlias = Mapping[Triple, float] | Mapping[Query, Mapping[str, float]]
TripleScorer: TypeAlias = Callable[[Triple], float]


def auroc(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Return AUROC using average ranks so ties are handled deterministically."""

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return math.nan
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positives].sum())
    auc = (rank_sum - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return float(auc)


def auprc(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Return average precision, the primary metric for sparse positives."""

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    true_positive_count = np.cumsum(ordered)
    positions = np.arange(1, len(ordered) + 1)
    precision_at_k = true_positive_count / positions
    return float((precision_at_k * ordered).sum() / positives)


def hits_at_k(ranks: Sequence[float] | np.ndarray, k: int) -> float:
    """Return the fraction of true tails ranked at or above ``k``."""

    values = np.asarray(ranks, dtype=float)
    if values.size == 0:
        return math.nan
    return float((values <= k).mean())


def mrr(ranks: Sequence[float] | np.ndarray) -> float:
    """Return mean reciprocal rank."""

    values = np.asarray(ranks, dtype=float)
    if values.size == 0:
        return math.nan
    return float((1.0 / values).mean())


def filtered_ranks(
    scores: ScoreMap,
    true_triples: Sequence[Triple],
    all_known: Sequence[Triple] | set[Triple],
    *,
    filtered: bool = True,
) -> np.ndarray:
    """Rank true tails, optionally removing other known true triples first."""

    known = set(all_known)
    ranks: list[float] = []
    for triple in true_triples:
        head, relation, true_tail = triple
        candidates = dict(_candidate_scores(scores, head, relation))
        if true_tail not in candidates:
            msg = f"Missing true tail score for {triple}"
            raise KeyError(msg)
        if filtered:
            for candidate_tail in list(candidates):
                if (
                    candidate_tail != true_tail
                    and (head, relation, candidate_tail) in known
                ):
                    del candidates[candidate_tail]
        true_score = candidates[true_tail]
        greater = sum(score > true_score for score in candidates.values())
        ties = sum(
            score == true_score
            for tail, score in candidates.items()
            if tail != true_tail
        )
        ranks.append(1.0 + greater + 0.5 * ties)
    return np.asarray(ranks, dtype=float)


def sample_negative_triples(
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    tail_candidates: Mapping[Query, Sequence[str]],
    negatives_per_positive: int = 1_000,
    seed: int = 13,
) -> list[Triple]:
    """Sample negatives per positive, or all eligible tails if fewer exist."""

    rng = random.Random(seed)
    known = set(all_known)
    negatives: list[Triple] = []
    used: set[Triple] = set()
    for head, relation, true_tail in positives:
        candidates = [
            tail
            for tail in tail_candidates[(head, relation)]
            if tail != true_tail and (head, relation, tail) not in known
        ]
        rng.shuffle(candidates)
        for tail in candidates[:negatives_per_positive]:
            triple = (head, relation, tail)
            if triple not in used:
                negatives.append(triple)
                used.add(triple)
    return negatives


def build_candidate_score_map(
    scorer: TripleScorer,
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    tail_candidates: Mapping[Query, Sequence[str]],
    negatives_per_positive: int = 1_000,
    seed: int = 13,
) -> dict[Query, dict[str, float]]:
    """Build query -> tail score maps for OGB-style positive-tail ranking."""

    known = set(all_known)
    rng = random.Random(seed)
    score_map: dict[Query, dict[str, float]] = {}
    for head, relation, true_tail in positives:
        query = (head, relation)
        eligible = [
            tail
            for tail in tail_candidates[query]
            if tail == true_tail or (head, relation, tail) not in known
        ]
        other_tails = [tail for tail in eligible if tail != true_tail]
        rng.shuffle(other_tails)
        sampled = [true_tail, *other_tails[:negatives_per_positive]]
        query_scores = score_map.setdefault(query, {})
        for tail in sorted(set(sampled)):
            query_scores[tail] = float(scorer((head, relation, tail)))
    return score_map


def evaluate_binary_and_ranking(
    scorer: TripleScorer,
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    tail_candidates: Mapping[Query, Sequence[str]],
    negatives_per_positive: int = 1_000,
    seed: int = 13,
    ks: Sequence[int] = (1, 3, 10),
) -> dict[str, object]:
    """Evaluate a triple scorer with sampled negatives and raw/filtered ranking."""

    negatives = sample_negative_triples(
        positives,
        all_known=all_known,
        tail_candidates=tail_candidates,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    triples = [*positives, *negatives]
    labels = [1] * len(positives) + [0] * len(negatives)
    scores = [float(scorer(triple)) for triple in triples]
    candidate_scores = build_candidate_score_map(
        scorer,
        positives,
        all_known=all_known,
        tail_candidates=tail_candidates,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    raw_ranks = filtered_ranks(candidate_scores, positives, set(), filtered=False)
    filt_ranks = filtered_ranks(candidate_scores, positives, all_known, filtered=True)

    raw = _rank_metrics(raw_ranks, ks)
    filtered_metrics = _rank_metrics(filt_ranks, ks)
    return {
        "primary_metric": "AUPRC",
        "auroc_note": "AUROC is optimistic under sparse-link imbalance.",
        "AUROC": auroc(labels, scores),
        "AUPRC": auprc(labels, scores),
        "raw": raw,
        "filtered": filtered_metrics,
        "raw_ranks": raw_ranks.tolist(),
        "filtered_ranks": filt_ranks.tolist(),
        "num_positives": len(positives),
        "num_negatives": len(negatives),
        "negatives_per_positive": negatives_per_positive,
    }


def _rank_metrics(ranks: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    metrics = {f"Hits@{k}": hits_at_k(ranks, k) for k in ks}
    metrics["MRR"] = mrr(ranks)
    return metrics


def _candidate_scores(
    scores: ScoreMap, head: str, relation: str
) -> Mapping[str, float]:
    query = (head, relation)
    raw_scores = cast(Mapping[Any, Any], scores)
    if query in raw_scores:
        return cast(Mapping[str, float], raw_scores[query])
    flat_scores = cast(Mapping[Triple, float], scores)
    return {
        tail: float(score)
        for (triple_head, triple_relation, tail), score in flat_scores.items()
        if triple_head == head and triple_relation == relation
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    return ranks
