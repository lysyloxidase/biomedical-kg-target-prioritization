"""Executable baseline suite for the split-first sample artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from kgtp.baselines.adjacency_svd import AdjacencySVDBaseline
from kgtp.baselines.common import evaluate_model
from kgtp.baselines.gradient_boosting import GradientBoostedTreesBaseline
from kgtp.baselines.kge import KGE_MODELS, KGEBaseline
from kgtp.baselines.logistic_regression import LogisticRegressionBaseline
from kgtp.baselines.matrix_factorization import MatrixFactorizationBaseline
from kgtp.baselines.mlp import FeatureMLPBaseline
from kgtp.baselines.node2vec import Node2VecBaseline
from kgtp.baselines.simple import (
    RandomScoreBaseline,
    SourceScoreBaseline,
    TargetPopularityBaseline,
    triples_from_pairs,
)
from kgtp.baselines.text_embeddings import (
    HashTextBaseline,
    optional_text_model_status,
)
from kgtp.eval.metrics import Query, Triple
from kgtp.hetero.feature_transformers import TrainGraphFeatureTransformer
from kgtp.hetero.split_protocol import TargetSplit
from kgtp.hetero.unlabeled_sampling import (
    DegreeMatchedUnlabeledSampler,
    HardUnlabeledSampler,
    RandomUnlabeledSampler,
    SamplingContext,
    SamplingResult,
    TargetProperties,
    mean_property_distance,
)

TARGET_RELATION = "associated_with"


def run_baseline_suite(
    *,
    nodes: pd.DataFrame,
    reference_edges: Mapping[str, pd.DataFrame],
    train_edges: Mapping[str, pd.DataFrame],
    split: TargetSplit,
    message_data: HeteroData,
    transformer: TrainGraphFeatureTransformer,
    models_dir: Path,
    metrics_dir: Path,
    sampling_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train, evaluate, and persist the supported baseline suite."""

    train_positive_pairs = sorted(
        _pairs(split.message_edges) | _positive_pairs(split.train_supervision)
    )
    validation_positive_pairs = sorted(_positive_pairs(split.validation_supervision))
    validation_negative_pairs = sorted(_negative_pairs(split.validation_supervision))
    test_positive_pairs = sorted(_positive_pairs(split.test_supervision))
    all_known_pairs = _pairs(split.full_known_positives)
    train_positives = triples_from_pairs(train_positive_pairs)
    validation_positives = triples_from_pairs(validation_positive_pairs)
    validation_negatives = triples_from_pairs(validation_negative_pairs)
    test_positives = triples_from_pairs(test_positive_pairs)
    all_known = set(triples_from_pairs(sorted(all_known_pairs)))
    gene_ids = tuple(str(value) for value in message_data["gene"].node_id)
    source_ids = tuple(sorted({source for source, _ in all_known_pairs}))
    tail_candidates: dict[Query, Sequence[str]] = {
        (source, TARGET_RELATION): gene_ids for source, _ in test_positive_pairs
    }

    context = build_sampling_context(
        train_edges=train_edges,
        transformer=transformer,
        source_ids=source_ids,
        target_ids=gene_ids,
        known_positive_pairs=all_known_pairs,
        train_positive_pairs=set(train_positive_pairs),
    )
    sampling_results = {
        "random": RandomUnlabeledSampler().sample(
            context,
            train_positive_pairs,
            num_samples=len(train_positive_pairs),
            seed=seed,
        ),
        "degree_matched": DegreeMatchedUnlabeledSampler().sample(
            context,
            train_positive_pairs,
            num_samples=len(train_positive_pairs),
            seed=seed,
        ),
        "hard": HardUnlabeledSampler().sample(
            context,
            train_positive_pairs,
            num_samples=len(train_positive_pairs),
            seed=seed,
        ),
    }
    _write_sampling_artifacts(
        sampling_results,
        train_positive_pairs=train_positive_pairs,
        context=context,
        output_dir=sampling_dir,
    )

    random_negatives = triples_from_pairs(sampling_results["random"].pairs)
    degree_negatives = triples_from_pairs(sampling_results["degree_matched"].pairs)
    hard_negatives = triples_from_pairs(sampling_results["hard"].pairs)
    node_features = _uniform_task_features(
        message_data,
        train_positive_pairs=train_positive_pairs,
    )
    graph_triples = _canonical_graph_triples(train_edges)
    descriptions = _descriptions(nodes)
    train_scores = _train_source_scores(
        reference_edges["disease_gene"],
        train_positive_pairs=set(train_positive_pairs),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    trained_models["random"] = RandomScoreBaseline(seed=seed)
    trained_models["degree_popularity"] = TargetPopularityBaseline().fit(
        {
            target: properties.degree
            for target, properties in context.target_properties.items()
        }
    )
    trained_models["source_score_only"] = SourceScoreBaseline().fit(train_scores)
    trained_models["logistic_regression"] = LogisticRegressionBaseline(epochs=200).fit(
        train_positives,
        random_negatives,
        node_features=node_features,
    )
    trained_models["logistic_regression_degree_matched"] = LogisticRegressionBaseline(
        epochs=200
    ).fit(
        train_positives,
        degree_negatives,
        node_features=node_features,
    )
    trained_models["logistic_regression_hard"] = LogisticRegressionBaseline(
        epochs=200
    ).fit(
        train_positives,
        hard_negatives,
        node_features=node_features,
    )
    trained_models["gradient_boosted_trees"] = GradientBoostedTreesBaseline().fit(
        train_positives,
        random_negatives,
        node_features=node_features,
    )
    trained_models["matrix_factorization"] = MatrixFactorizationBaseline(
        rank=8,
        epochs=150,
        seed=seed,
    ).fit(train_positives, random_negatives)
    trained_models["adjacency_svd"] = AdjacencySVDBaseline(dimension=16).fit(
        graph_triples
    )
    trained_models["node2vec"] = Node2VecBaseline(
        dimension=16,
        walk_length=8,
        walks_per_node=3,
        context_window=2,
        negative_samples=2,
        p=1.0,
        q=0.5,
        epochs=2,
        seed=seed,
    ).fit(graph_triples)
    trained_models["hash_text"] = HashTextBaseline(dim=64).fit(descriptions)
    trained_models["feature_mlp"] = FeatureMLPBaseline(
        hidden_channels=16,
        epochs=100,
        seed=seed,
    ).fit(
        train_positives,
        random_negatives,
        node_features=node_features,
        validation_positives=validation_positives,
        validation_negatives=validation_negatives,
    )

    kge_positives = [*graph_triples, *train_positives]
    for model_name in KGE_MODELS:
        key = model_name.lower()
        trained_models[key] = KGEBaseline(
            model_name=model_name,
            dimension=12,
            epochs=60,
            seed=seed,
        ).fit(
            kge_positives,
            random_negatives,
            validation_positives=validation_positives,
            validation_negatives=validation_negatives,
        )

    for name, model in trained_models.items():
        result = evaluate_model(
            cast(Any, model),
            test_positives,
            all_known=all_known,
            tail_candidates=tail_candidates,
            negatives_per_positive=16,
            seed=seed,
        )
        metrics[name] = result
        _write_json(metrics_dir / f"{name}.json", result)
        _save_model(name, model, models_dir)

    availability = optional_text_model_status()
    for _name, status in availability.items():
        if not status["available"]:
            status["status"] = "unavailable"
            status["reason"] = (
                f"Optional dependency '{status['dependency']}' is not installed; "
                "no fallback was executed"
            )
        else:
            status["status"] = "available_not_run"
            status["reason"] = "No model revision was pinned for the sample run"
    _write_json(models_dir / "availability.json", availability)

    summary = {
        "scope": "sample pipeline validation; not a scientific benchmark",
        "seed": seed,
        "input_provenance": {
            "full_reference_graph_hash": split.metadata["full_reference_graph_hash"],
            "train_message_graph_hash": split.metadata["train_message_graph_hash"],
            "node_index_map_hash": split.metadata["node_index_map_hash"],
            "feature_transformer_state_hash": transformer.state_hash(),
            "feature_fit_graph_hash": transformer.fitted_graph_hash,
        },
        "evaluation_protocol": {
            "test_positive_count": len(test_positives),
            "candidate_gene_count": len(gene_ids),
            "known_positive_filter_count": len(all_known),
            "negatives_per_positive": 16,
        },
        "training_unlabeled_strategy": {
            name: result.diagnostics for name, result in sampling_results.items()
        },
        "models": metrics,
        "optional_models": availability,
    }
    _write_json(metrics_dir / "results.json", summary)
    return summary


def build_sampling_context(
    *,
    train_edges: Mapping[str, pd.DataFrame],
    transformer: TrainGraphFeatureTransformer,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    known_positive_pairs: set[tuple[str, str]],
    train_positive_pairs: set[tuple[str, str]],
) -> SamplingContext:
    """Build degree properties and hard pools solely from train graph edges."""

    gene_features = transformer.features["gene"]
    annotation_counts: defaultdict[str, int] = defaultdict(int)
    pathway_counts: defaultdict[str, int] = defaultdict(int)
    gene_go: defaultdict[str, set[str]] = defaultdict(set)
    gene_pathways: defaultdict[str, set[str]] = defaultdict(set)
    ppi_neighbors: defaultdict[str, set[str]] = defaultdict(set)
    for row in train_edges["gene_go"].itertuples(index=False):
        annotation_counts[str(row.source_id)] += 1
        gene_go[str(row.source_id)].add(str(row.target_id))
    for row in train_edges["gene_pathway"].itertuples(index=False):
        pathway_counts[str(row.source_id)] += 1
        gene_pathways[str(row.source_id)].add(str(row.target_id))
    for row in train_edges["gene_gene"].itertuples(index=False):
        first = str(row.source_id)
        second = str(row.target_id)
        ppi_neighbors[first].add(second)
        ppi_neighbors[second].add(first)

    properties = {
        target: TargetProperties(
            degree=float(gene_features[target][0]),
            pagerank=float(gene_features[target][1]),
            annotation_count=float(annotation_counts[target]),
            pathway_count=float(pathway_counts[target]),
        )
        for target in target_ids
    }
    positives_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for source, target in train_positive_pairs:
        positives_by_source[source].add(target)

    hard_pools: dict[str, tuple[str, ...]] = {}
    for source in source_ids:
        positive_targets = positives_by_source[source]
        positive_pathways = set().union(
            *(gene_pathways[target] for target in positive_targets)
        )
        positive_go = set().union(*(gene_go[target] for target in positive_targets))
        ppi_pool = set().union(*(ppi_neighbors[target] for target in positive_targets))
        hard = {
            target
            for target in target_ids
            if target in ppi_pool
            or bool(gene_pathways[target] & positive_pathways)
            or bool(gene_go[target] & positive_go)
        }
        hard_pools[source] = tuple(sorted(hard - positive_targets))
    return SamplingContext(
        source_ids=tuple(source_ids),
        target_ids=tuple(target_ids),
        known_positive_pairs=frozenset(known_positive_pairs),
        target_properties=properties,
        hard_candidates_by_source=hard_pools,
    )


def _write_sampling_artifacts(
    results: Mapping[str, SamplingResult],
    *,
    train_positive_pairs: Sequence[tuple[str, str]],
    context: SamplingContext,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        frame = pd.DataFrame(
            [
                {
                    "source_id": source,
                    "target_id": target,
                    "strategy": result.strategy,
                    "label_semantics": "unlabeled_not_confirmed_negative",
                }
                for source, target in result.pairs
            ]
        )
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        diagnostics = dict(result.diagnostics)
        diagnostics["mean_property_distance_to_positives"] = mean_property_distance(
            train_positive_pairs,
            result.pairs,
            context.target_properties,
        )
        _write_json(output_dir / f"{name}.json", diagnostics)


def _uniform_task_features(
    data: HeteroData,
    *,
    train_positive_pairs: Sequence[tuple[str, str]],
) -> dict[str, list[float]]:
    gene_ids = [str(value) for value in data["gene"].node_id]
    gene_features = {
        gene_id: data["gene"].x[index].tolist()
        for index, gene_id in enumerate(gene_ids)
    }
    width = data["gene"].x.size(1)
    positive_targets = [target for _, target in train_positive_pairs]
    source_profile = np.mean(
        [gene_features[target] for target in positive_targets],
        axis=0,
    ).tolist()
    features = dict(gene_features)
    for source in sorted({source for source, _ in train_positive_pairs}):
        features[source] = source_profile[:width]
    return features


def _canonical_graph_triples(
    edge_tables: Mapping[str, pd.DataFrame],
) -> list[Triple]:
    triples: list[Triple] = []
    for edges in edge_tables.values():
        triples.extend(
            (
                str(row.source_id),
                str(row.edge_type),
                str(row.target_id),
            )
            for row in edges.itertuples(index=False)
        )
    return sorted(set(triples))


def _descriptions(nodes: pd.DataFrame) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for row in nodes.itertuples(index=False):
        values = [
            str(value)
            for value in (
                getattr(row, "label", ""),
                getattr(row, "symbol", ""),
                getattr(row, "description", ""),
            )
            if value is not None and str(value) != "nan"
        ]
        descriptions[str(row.node_id)] = " ".join(values) or str(row.node_id)
    return descriptions


def _train_source_scores(
    disease_gene: pd.DataFrame,
    *,
    train_positive_pairs: set[tuple[str, str]],
) -> dict[Triple, float]:
    return {
        (str(row.source_id), TARGET_RELATION, str(row.target_id)): float(str(row.score))
        for row in disease_gene.itertuples(index=False)
        if (str(row.source_id), str(row.target_id)) in train_positive_pairs
    }


def _save_model(name: str, model: object, output_dir: Path) -> None:
    metadata: dict[str, Any] = {"model": name}
    if isinstance(model, LogisticRegressionBaseline):
        np.savez(
            output_dir / f"{name}.npz",
            weights=model.weights,
            bias=np.asarray([model.bias]),
        )
    elif isinstance(model, MatrixFactorizationBaseline):
        np.savez(
            output_dir / f"{name}.npz",
            source_factors=model.source_factors,
            tail_factors=model.tail_factors,
        )
        metadata.update(
            {
                "source_index": model.source_index,
                "tail_index": model.tail_index,
                "rank": model.rank,
            }
        )
    elif isinstance(model, AdjacencySVDBaseline | Node2VecBaseline):
        ids = sorted(model.embeddings)
        matrix = np.vstack([model.embeddings[node_id] for node_id in ids])
        np.savez(output_dir / f"{name}.npz", ids=np.asarray(ids), embeddings=matrix)
        if isinstance(model, Node2VecBaseline):
            metadata["hyperparameters"] = model.hyperparameters()
    elif isinstance(model, HashTextBaseline):
        ids = sorted(model.embeddings)
        matrix = np.vstack([model.embeddings[node_id] for node_id in ids])
        np.savez(output_dir / f"{name}.npz", ids=np.asarray(ids), embeddings=matrix)
        metadata["dimension"] = model.dim
    elif isinstance(model, KGEBaseline):
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        metadata.update(model.metadata())
    elif isinstance(model, FeatureMLPBaseline):
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        metadata.update(
            {
                "hidden_channels": model.hidden_channels,
                "epochs": model.epochs,
                "seed": model.seed,
            }
        )
    elif isinstance(model, GradientBoostedTreesBaseline):
        metadata["stumps"] = [
            {
                "feature": stump.feature,
                "threshold": stump.threshold,
                "left_value": stump.left_value,
                "right_value": stump.right_value,
            }
            for stump in model.stumps
        ]
        metadata["bias"] = model.bias
    elif isinstance(model, SourceScoreBaseline):
        metadata["train_scores"] = {
            "\t".join(triple): score for triple, score in sorted(model.scores.items())
        }
    elif isinstance(model, TargetPopularityBaseline):
        metadata["target_scores"] = model.target_scores
    elif isinstance(model, RandomScoreBaseline):
        metadata["seed"] = model.seed
    _write_json(output_dir / f"{name}.json", metadata)


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(source), str(target))
        for source, target in zip(frame["source_id"], frame["target_id"], strict=True)
    }


def _positive_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return _pairs(frame.loc[frame["label"] == 1])


def _negative_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return _pairs(frame.loc[frame["label"] == 0])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
