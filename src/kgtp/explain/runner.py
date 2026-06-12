"""Validated explainability workflow and candidate evidence cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from kgtp.artifacts import ValidatedArtifacts
from kgtp.explain.explainer import DISEASE_GENE_EDGE, TargetExplainer


def rank_all_genes(
    artifacts: ValidatedArtifacts, disease_idx: int
) -> list[dict[str, Any]]:
    """Rank every eligible gene with the validated trained model."""
    data = artifacts.data
    model = artifacts.model
    gene_ids = [str(value) for value in data["gene"].node_id]
    disease_ids = [str(value) for value in data["disease"].node_id]
    if disease_idx < 0 or disease_idx >= len(disease_ids):
        raise ValueError(f"Unknown disease index: {disease_idx}")
    edge_index = torch.stack(
        (
            torch.full((len(gene_ids),), disease_idx, dtype=torch.long),
            torch.arange(len(gene_ids), dtype=torch.long),
        )
    )
    with torch.no_grad():
        embeddings = cast(Any, model).encode(data.x_dict, _edge_index_dict(data))
        scores = cast(Any, model).decode(
            embeddings,
            DISEASE_GENE_EDGE,
            edge_index,
        )
    order = sorted(
        range(len(gene_ids)),
        key=lambda index: (-float(scores[index].item()), gene_ids[index]),
    )
    return [
        {
            "rank": rank,
            "disease_idx": disease_idx,
            "disease_id": disease_ids[disease_idx],
            "gene_idx": gene_idx,
            "gene_id": gene_ids[gene_idx],
            "model_score": float(scores[gene_idx].item()),
            "model_probability": float(torch.sigmoid(scores[gene_idx]).item()),
        }
        for rank, gene_idx in enumerate(order, start=1)
    ]


def run_validated_explanations(
    artifacts: ValidatedArtifacts,
    output_dir: Path,
    *,
    disease_id: str | None = None,
    integration_steps: int = 8,
) -> dict[str, Any]:
    """Generate model-scored case explanations and evidence cards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = _graph_tables(artifacts)
    _attach_labels(artifacts, tables["nodes"])
    disease_ids = [str(value) for value in artifacts.data["disease"].node_id]
    selected_disease = disease_id or disease_ids[0]
    if selected_disease not in disease_ids:
        raise ValueError(f"Unknown disease ID: {selected_disease}")
    disease_idx = disease_ids.index(selected_disease)

    ranking = rank_all_genes(artifacts, disease_idx)
    by_gene = {str(row["gene_id"]): row for row in ranking}
    categories = _select_cases(artifacts, tables, selected_disease, ranking)
    explainer = TargetExplainer(
        artifacts.model,
        artifacts.data,
        edge_type=DISEASE_GENE_EDGE,
        integration_steps=integration_steps,
        use_pyg_captum=False,
    )

    explanations: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for category, gene_id in categories:
        ranked = by_gene[gene_id]
        explanation = explainer.explain_link(disease_idx, int(ranked["gene_idx"]))
        subgraph = explainer.explanatory_subgraph(explanation)
        explanations.append(
            {
                "category": category,
                "gene_id": gene_id,
                "rank": ranked["rank"],
                "model_score": ranked["model_score"],
                "attribution": subgraph,
                "warning": (
                    "Integrated Gradients and edge occlusion are model attributions, "
                    "not causal explanations."
                ),
            }
        )
        cards.append(
            _evidence_card(
                category,
                ranked,
                tables,
                artifacts,
            )
        )

    metadata = _prediction_metadata(artifacts)
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "cases": explanations,
    }
    _write_json(output_dir / "ranking.json", {"metadata": metadata, "ranking": ranking})
    _write_json(output_dir / "explanations.json", payload)
    _write_json(
        output_dir / "evidence_cards.json",
        {"metadata": metadata, "evidence_cards": cards},
    )
    return {
        "ranking_count": len(ranking),
        "explanation_count": len(explanations),
        "output_dir": str(output_dir),
        "categories": [category for category, _ in categories],
    }


def _select_cases(
    artifacts: ValidatedArtifacts,
    tables: dict[str, pd.DataFrame],
    disease_id: str,
    ranking: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    ranked_ids = [str(row["gene_id"]) for row in ranking]
    full_known = set(
        tables["disease_gene"]
        .loc[tables["disease_gene"]["source_id"] == disease_id, "target_id"]
        .astype(str)
    )
    nodes = tables["nodes"]
    symbols = {
        str(row.node_id): str(row.symbol)
        for row in nodes.loc[nodes["node_type"] == "Gene"].itertuples()
    }
    preferred = next(
        (
            gene_id
            for symbol in ("GDF5", "MMP13")
            for gene_id, candidate_symbol in symbols.items()
            if candidate_symbol == symbol and gene_id in full_known
        ),
        None,
    )
    known = preferred or next(
        gene_id for gene_id in ranked_ids if gene_id in full_known
    )

    test = pd.read_parquet(
        artifacts.paths.split_metadata.parent / "supervision" / "test.parquet"
    )
    held_out = set(
        test.loc[
            (test["source_id"] == disease_id) & (test["label"] == 1),
            "target_id",
        ].astype(str)
    )
    ranked_held_out = [gene_id for gene_id in ranked_ids if gene_id in held_out]
    if not ranked_held_out:
        raise ValueError("No held-out positive is available for explanation")

    unlabeled_path = (
        artifacts.paths.split_metadata.parent / "negative_sampling" / "hard.parquet"
    )
    unlabeled = set(pd.read_parquet(unlabeled_path)["target_id"].astype(str))
    ranked_unlabeled = [gene_id for gene_id in ranked_ids if gene_id in unlabeled]
    if not ranked_unlabeled:
        raise ValueError("No sampled-unlabeled candidate is available for explanation")

    novel = next(
        (gene_id for gene_id in ranked_ids if gene_id not in full_known),
        None,
    )
    if novel is None:
        raise ValueError("No model-scored novel unlabeled hypothesis is available")

    return [
        ("known_target_sanity_check", known),
        ("recovered_held_out_positive", ranked_held_out[0]),
        ("false_negative_held_out_positive", ranked_held_out[-1]),
        ("false_positive_proxy_sampled_unlabeled", ranked_unlabeled[0]),
        ("novel_unlabeled_hypothesis", novel),
    ]


def _evidence_card(
    category: str,
    ranked: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    artifacts: ValidatedArtifacts,
) -> dict[str, Any]:
    gene_id = str(ranked["gene_id"])
    nodes = tables["nodes"].set_index("node_id", drop=False)
    gene_row = nodes.loc[gene_id]
    pathways = tables["gene_pathway"].loc[
        tables["gene_pathway"]["source_id"] == gene_id
    ]
    ppi = tables["gene_gene"].loc[
        (tables["gene_gene"]["source_id"] == gene_id)
        | (tables["gene_gene"]["target_id"] == gene_id)
    ]
    go_terms = tables["gene_go"].loc[tables["gene_go"]["source_id"] == gene_id]
    drugs = tables["drug_gene"].loc[tables["drug_gene"]["target_id"] == gene_id]

    ppi_neighbors = []
    for row in ppi.itertuples():
        neighbor = str(row.target_id if row.source_id == gene_id else row.source_id)
        ppi_neighbors.append(
            {
                "gene_id": neighbor,
                "symbol": _node_value(nodes, neighbor, "symbol"),
                "source": str(row.source),
                "score": float(str(row.score)),
            }
        )
    return {
        "rank": int(ranked["rank"]),
        "gene_id": gene_id,
        "gene_symbol": str(gene_row["symbol"]),
        "model_score": float(ranked["model_score"]),
        "model_probability": float(ranked["model_probability"]),
        "known_or_held_out_status": category,
        "pathways": [
            {
                "pathway_id": str(row.target_id),
                "name": str(row.pathway_name),
                "source": str(row.source),
            }
            for row in pathways.itertuples()
        ],
        "ppi_neighbors": ppi_neighbors,
        "go_terms": [
            {
                "go_id": str(row.target_id),
                "label": _node_value(nodes, str(row.target_id), "label"),
                "evidence_code": str(row.evidence_code),
                "source": str(row.source),
            }
            for row in go_terms.itertuples()
        ],
        "drug_information": [
            {
                "drug_id": str(row.source_id),
                "drug_name": _node_value(nodes, str(row.source_id), "label"),
                "action_type": str(row.action_type),
                "mechanism_of_action": str(row.mechanism_of_action),
                "source": str(row.source),
            }
            for row in drugs.itertuples()
        ],
        "evidence_provenance": sorted(
            {
                str(value)
                for frame in (pathways, ppi, go_terms, drugs)
                for value in frame.get("source", pd.Series(dtype=str)).tolist()
            }
        ),
        "uncertainty": {
            "probability_like_score": float(ranked["model_probability"]),
            "calibrated": False,
            "predictive_interval": None,
            "limitation": "Single-checkpoint attribution; predictive uncertainty is not estimated.",
        },
        "checkpoint_sha256": artifacts.checkpoint_sha256,
        "warning": (
            "Computational hypothesis only. The score and model attributions do not "
            "establish biological validity or causality."
        ),
    }


def _graph_tables(artifacts: ValidatedArtifacts) -> dict[str, pd.DataFrame]:
    records = artifacts.graph_manifest["outputs"]
    paths = {
        Path(record["path"]).name: Path(record["path"])
        for record in records
        if str(record["path"]).endswith(".parquet")
    }
    edge_paths = {
        Path(record["path"]).stem: Path(record["path"])
        for record in records
        if "/edges/" in str(record["path"]).replace("\\", "/")
    }
    required = {
        "nodes": paths.get("nodes.parquet"),
        "disease_gene": edge_paths.get("disease_gene"),
        "gene_gene": edge_paths.get("gene_gene"),
        "gene_pathway": edge_paths.get("gene_pathway"),
        "gene_go": edge_paths.get("gene_go"),
        "drug_gene": edge_paths.get("drug_gene"),
    }
    if any(path is None for path in required.values()):
        raise ValueError("Graph manifest does not declare all evidence-card tables")
    return {name: pd.read_parquet(cast(Path, path)) for name, path in required.items()}


def _attach_labels(artifacts: ValidatedArtifacts, nodes: pd.DataFrame) -> None:
    type_map = {
        "Disease": "disease",
        "Drug": "drug",
        "Gene": "gene",
        "GOTerm": "go_term",
        "Pathway": "pathway",
    }
    indexed = nodes.set_index("node_id", drop=False)
    for source_type, node_type in type_map.items():
        ids = [str(value) for value in artifacts.data[node_type].node_id]
        subset = nodes.loc[nodes["node_type"] == source_type]
        if len(subset) != len(ids):
            continue
        for column in ("label", "symbol"):
            values = [
                str(indexed.loc[node_id, column]) if node_id in indexed.index else ""
                for node_id in ids
            ]
            setattr(artifacts.data[node_type], column, values)


def _prediction_metadata(artifacts: ValidatedArtifacts) -> dict[str, Any]:
    metrics = artifacts.validation_metrics.get("metrics", {})
    return {
        "model_name": artifacts.model_config["model_name"],
        "model_version": artifacts.model_config["model_version"],
        "run_id": artifacts.run_manifest["run_id"],
        "checkpoint_sha256": artifacts.checkpoint_sha256,
        "dataset_id": artifacts.dataset_manifest["dataset_id"],
        "dataset_manifest_sha256": artifacts.dataset_manifest_sha256,
        "split_id": artifacts.artifact_metadata["split_hash"],
        "trained": True,
        "validation_metrics": metrics,
        "candidate_protocol": "full_candidate_all_eligible_genes",
        "hypothesis_only": True,
        "warnings": [
            "Scores are computational hypotheses from a small sample dataset.",
            "Model attributions are not causal explanations.",
        ],
    }


def _node_value(nodes: pd.DataFrame, node_id: str, column: str) -> str:
    if node_id not in nodes.index:
        return ""
    value = nodes.loc[node_id, column]
    return str(value)


def _edge_index_dict(data: Any) -> dict[tuple[str, str, str], torch.Tensor]:
    return {
        cast(tuple[str, str, str], edge_type): data[edge_type].edge_index
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
