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
    """Return tie-aware average precision for sparse positives."""

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    true_positives = 0
    seen = 0
    average_precision = 0.0
    start = 0
    while start < len(ordered_scores):
        end = start + 1
        while (
            end < len(ordered_scores) and ordered_scores[end] == ordered_scores[start]
        ):
            end += 1
        group_positives = int(ordered_labels[start:end].sum())
        true_positives += group_positives
        seen = end
        precision = true_positives / seen
        recall_increment = group_positives / positives
        average_precision += precision * recall_increment
        start = end
    return float(average_precision)


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


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""

    labels = np.asarray(y_true, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if labels.size == 0:
        return math.nan
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (probs >= lower) & (
            probs <= upper if index == bins - 1 else probs < upper
        )
        if not mask.any():
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        error += float(mask.mean()) * abs(accuracy - confidence)
    return error


def brier_score(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    """Return mean squared probability error."""

    labels = np.asarray(y_true, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if labels.size == 0:
        return math.nan
    return float(np.mean((probs - labels) ** 2))


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
    """Sample unobserved tails, excluding every registered positive."""

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
    ks: Sequence[int] = (1, 3, 10, 50),
) -> dict[str, object]:
    """Evaluate a scorer with sampled unlabeled pairs and filtered ranking."""

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
    filtered_metrics.update(
        _query_topk_metrics(
            candidate_scores,
            positives,
            all_known=all_known,
            ks=ks,
        )
    )
    prevalence = len(positives) / len(triples) if triples else math.nan
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
        "unlabeled_count": len(negatives),
        "candidate_prevalence": prevalence,
        "candidate_prevalence_warning": (
            "AUPRC is prevalence-dependent; compare only runs using the same "
            "candidate protocol and prevalence."
        ),
        "unlabeled_per_positive": negatives_per_positive,
        "label_semantics": {"1": "positive", "0": "unlabeled"},
    }


def evaluate_full_candidate(
    scorer: TripleScorer,
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    tail_candidates: Mapping[Query, Sequence[str]],
    probability_scorer: TripleScorer | None = None,
    ks: Sequence[int] = (1, 3, 10, 50),
) -> dict[str, object]:
    """Evaluate all eligible tails for every held-out query."""

    known = set(all_known)
    positive_set = set(positives)
    queries = sorted({(head, relation) for head, relation, _ in positives})
    candidate_scores: dict[Query, dict[str, float]] = {}
    triples: list[Triple] = []
    labels: list[int] = []
    for query in queries:
        head, relation = query
        relevant = {
            tail
            for positive_head, positive_relation, tail in positive_set
            if (positive_head, positive_relation) == query
        }
        eligible = [
            tail
            for tail in tail_candidates[query]
            if tail in relevant or (head, relation, tail) not in known
        ]
        query_scores = {
            tail: float(scorer((head, relation, tail)))
            for tail in sorted(set(eligible))
        }
        candidate_scores[query] = query_scores
        for tail in sorted(query_scores):
            triples.append((head, relation, tail))
            labels.append(int(tail in relevant))
    scores = [
        candidate_scores[(head, relation)][tail] for head, relation, tail in triples
    ]
    raw_ranks = filtered_ranks(candidate_scores, positives, set(), filtered=False)
    filtered_values = filtered_ranks(
        candidate_scores,
        positives,
        all_known,
        filtered=True,
    )
    prevalence = sum(labels) / len(labels) if labels else math.nan
    result: dict[str, object] = {
        "protocol": "full_candidate_all_eligible_tails",
        "label_semantics": {"1": "positive", "0": "unlabeled"},
        "AUROC": auroc(labels, scores),
        "AUPRC": auprc(labels, scores),
        "candidate_prevalence": prevalence,
        "candidate_count": len(labels),
        "positive_count": sum(labels),
        "unlabeled_count": len(labels) - sum(labels),
        "candidate_prevalence_warning": (
            "AUPRC is prevalence-dependent; do not compare it with sampled-set "
            "AUPRC without accounting for candidate prevalence."
        ),
        "raw": _rank_metrics(raw_ranks, ks),
        "filtered": {
            **_rank_metrics(filtered_values, ks),
            **_query_topk_metrics(
                candidate_scores,
                positives,
                all_known=all_known,
                ks=ks,
            ),
        },
        "raw_ranks": raw_ranks.tolist(),
        "filtered_ranks": filtered_values.tolist(),
    }
    if probability_scorer is not None:
        probabilities = [float(probability_scorer(triple)) for triple in triples]
        result["calibration"] = {
            "Brier": brier_score(labels, probabilities),
            "ECE": expected_calibration_error(labels, probabilities),
        }
    return result


def evaluate_sampled_unlabeled(
    scorer: TripleScorer,
    positives: Sequence[Triple],
    unlabeled: Sequence[Triple],
    *,
    strategy: str,
    probability_scorer: TripleScorer | None = None,
) -> dict[str, object]:
    """Evaluate positives against an explicit sampled-unlabeled set."""

    positive_unique = sorted(set(positives))
    unlabeled_unique = sorted(set(unlabeled) - set(positive_unique))
    triples = [*positive_unique, *unlabeled_unique]
    labels = [1] * len(positive_unique) + [0] * len(unlabeled_unique)
    scores = [float(scorer(triple)) for triple in triples]
    prevalence = len(positive_unique) / len(triples) if triples else math.nan
    result: dict[str, object] = {
        "protocol": f"sampled_unlabeled_{strategy}",
        "label_semantics": {"1": "positive", "0": "unlabeled"},
        "AUROC": auroc(labels, scores),
        "AUPRC": auprc(labels, scores),
        "positive_count": len(positive_unique),
        "unlabeled_count": len(unlabeled_unique),
        "candidate_prevalence": prevalence,
        "candidate_prevalence_warning": (
            "AUPRC is specific to this sampled-unlabeled prevalence and is not "
            "directly comparable with other candidate protocols."
        ),
    }
    if probability_scorer is not None:
        probabilities = [float(probability_scorer(triple)) for triple in triples]
        result["calibration"] = {
            "Brier": brier_score(labels, probabilities),
            "ECE": expected_calibration_error(labels, probabilities),
        }
    return result


def _rank_metrics(ranks: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    metrics = {f"Hits@{k}": hits_at_k(ranks, k) for k in ks}
    metrics["MRR"] = mrr(ranks)
    return metrics


def _query_topk_metrics(
    score_map: Mapping[Query, Mapping[str, float]],
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    ks: Sequence[int],
) -> dict[str, float]:
    known = set(all_known)
    positives_by_query: dict[Query, set[str]] = {}
    for head, relation, tail in positives:
        positives_by_query.setdefault((head, relation), set()).add(tail)
    collected: dict[str, list[float]] = {
        metric: []
        for k in ks
        for metric in (f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", f"EF@{k}")
    }
    for query, relevant in positives_by_query.items():
        head, relation = query
        candidates = {
            tail: score
            for tail, score in score_map[query].items()
            if tail in relevant or (head, relation, tail) not in known
        }
        ranked = sorted(candidates, key=lambda tail: (-candidates[tail], tail))
        prevalence = len(relevant) / len(ranked) if ranked else math.nan
        for k in ks:
            top = ranked[:k]
            hits = sum(tail in relevant for tail in top)
            denominator = min(k, len(ranked))
            precision = hits / denominator if denominator else math.nan
            recall = hits / len(relevant) if relevant else math.nan
            dcg = sum(
                1.0 / math.log2(index + 2)
                for index, tail in enumerate(top)
                if tail in relevant
            )
            ideal_hits = min(len(relevant), denominator)
            ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
            ndcg = dcg / ideal if ideal else math.nan
            ef = (
                precision / prevalence
                if prevalence and not math.isnan(prevalence)
                else math.nan
            )
            collected[f"Precision@{k}"].append(precision)
            collected[f"Recall@{k}"].append(recall)
            collected[f"NDCG@{k}"].append(ndcg)
            collected[f"EF@{k}"].append(ef)
    return {
        metric: float(np.mean(values)) if values else math.nan
        for metric, values in collected.items()
    }


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
