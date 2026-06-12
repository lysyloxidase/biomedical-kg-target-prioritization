"""Command-line interface for the benchmark pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from kgtp.ablation.common import read_report
from kgtp.ablation.tables import write_ablation_tables
from kgtp.artifacts import (
    ArtifactPaths,
    ArtifactValidationError,
    validate_and_load_artifacts,
)
from kgtp.config import load_settings, load_yaml
from kgtp.data.build_graph import assert_canonical_edge_types
from kgtp.data.common import read_table
from kgtp.data.opentargets import fetch_oa_target_count
from kgtp.explain.runner import run_validated_explanations
from kgtp.hetero.build_heterodata import build_heterodata
from kgtp.hetero.splits import disjoint_random_link_split, load_splits, save_splits
from kgtp.kg.neo4j_loader import Neo4jConfig, load_graph
from kgtp.kg.statistics import graph_statistics_table
from kgtp.models.train import TrainingConfig, train_one_seed
from kgtp.pipeline.sample import (
    SamplePaths,
    assemble_sample,
    build_sample_features,
    clean_sample,
    evaluate_sample,
    normalize_sample,
    prepare_sample,
    report_sample,
    reproduce_small,
    split_sample,
    train_sample_baselines,
    train_sample_gnn,
)
from kgtp.smoke import run_smoke_train

app = typer.Typer(help="OA-centric heterogeneous KG benchmark utilities.")

SOURCES_PATH_OPTION = typer.Option(Path("configs/sources.yaml"))
FETCH_LIVE_COUNT_OPTION = typer.Option(
    False,
    help="Query Open Targets GraphQL for the live EFO_0004616 target count.",
)
PROCESSED_DIR_OPTION = typer.Option(Path("data/processed"))
CONFIG_PATH_OPTION = typer.Option(Path("configs/config.yaml"))
HETERODATA_DIR_OPTION = typer.Option(Path("data/processed/heterodata"))
HETERODATA_PATH_OPTION = typer.Option(Path("data/processed/heterodata/heterodata.pt"))
SPLITS_DIR_OPTION = typer.Option(Path("data/processed/splits"))
SEED_OPTION = typer.Option(13)
MODEL_OUTPUT_DIR_OPTION = typer.Option(Path("reports/models/hgt_seed13"))
MODEL_NAME_OPTION = typer.Option("hgt")
MAX_EPOCHS_OPTION = typer.Option(50)
ABLATION_REPORTS_DIR_OPTION = typer.Option(Path("reports/ablations"))
ABLATION_TABLES_DIR_OPTION = typer.Option(Path("reports/ablation_tables"))
EXPLAIN_OUTPUT_DIR_OPTION = typer.Option(Path("reports/figures"))
EXPLAIN_CHECKPOINT_OPTION = typer.Option(..., help="Trained best-checkpoint artifact.")
EXPLAIN_MODEL_CONFIG_OPTION = typer.Option(..., help="Saved model configuration JSON.")
EXPLAIN_GRAPH_OPTION = typer.Option(..., help="Train-message PyG HeteroData artifact.")
EXPLAIN_DATASET_MANIFEST_OPTION = typer.Option(..., help="Dataset manifest JSON.")
EXPLAIN_GRAPH_MANIFEST_OPTION = typer.Option(..., help="Full reference graph manifest.")
EXPLAIN_FEATURE_MANIFEST_OPTION = typer.Option(
    ..., help="Train-fitted feature manifest."
)
EXPLAIN_NODE_INDEX_MAP_OPTION = typer.Option(..., help="Node-index map JSON.")
EXPLAIN_SPLIT_METADATA_OPTION = typer.Option(..., help="Validated split metadata JSON.")
EXPLAIN_VALIDATION_METRICS_OPTION = typer.Option(
    ..., help="Saved validation metrics JSON."
)
EXPLAIN_RUN_MANIFEST_OPTION = typer.Option(..., help="Completed run manifest JSON.")
EXPLAIN_VALIDATED_OUTPUT_OPTION = typer.Option(
    Path("artifacts/sample/report/explanations"),
    help="Explanation output directory.",
)
SMOKE_OUTPUT_DIR_OPTION = typer.Option(Path("reports/smoke_train"))
SAMPLE_DIR_OPTION = typer.Option(Path("data/sample"))
SAMPLE_ARTIFACT_ROOT_OPTION = typer.Option(Path("artifacts/sample"))
SAMPLE_MAX_EPOCHS_OPTION = typer.Option(5, min=1)


def _sample_paths(sample_dir: Path, artifact_root: Path) -> SamplePaths:
    return SamplePaths(sample_dir=sample_dir, root=artifact_root)


@app.command("prepare-sample")
def prepare_sample_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Validate the checked-in redistributable sample snapshot."""

    prepare_sample(_sample_paths(sample_dir, artifact_root))


@app.command("normalize")
def normalize_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Normalize the sample identifiers into deterministic Parquet tables."""

    normalize_sample(_sample_paths(sample_dir, artifact_root))


@app.command("assemble")
def assemble_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Assemble canonical graph tables from normalized sample data."""

    assemble_sample(_sample_paths(sample_dir, artifact_root))


@app.command("features")
def features_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Build the sample PyG HeteroData and feature tensors."""

    build_sample_features(_sample_paths(sample_dir, artifact_root))


@app.command("split")
def split_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Create deterministic sample disease-gene link splits."""

    split_sample(_sample_paths(sample_dir, artifact_root), seed=seed)


@app.command("train-baselines")
def train_baselines_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Fit the executable small-sample baseline models."""

    train_sample_baselines(_sample_paths(sample_dir, artifact_root), seed=seed)


@app.command("train-gnn")
def train_gnn_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
    seed: int = SEED_OPTION,
    max_epochs: int = SAMPLE_MAX_EPOCHS_OPTION,
) -> None:
    """Train four GNN families across five fixed sample seeds."""

    train_sample_gnn(
        _sample_paths(sample_dir, artifact_root),
        seed=seed,
        max_epochs=max_epochs,
    )


@app.command("evaluate")
def evaluate_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Verify reloaded baseline and multi-seed GNN results."""

    evaluate_sample(_sample_paths(sample_dir, artifact_root), seed=seed)


@app.command("report")
def report_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Write machine-readable and Markdown sample reports."""

    report_sample(_sample_paths(sample_dir, artifact_root))


@app.command("reproduce-small")
def reproduce_small_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
    seed: int = SEED_OPTION,
    max_epochs: int = SAMPLE_MAX_EPOCHS_OPTION,
) -> None:
    """Run the complete checked-in sample pipeline."""

    reproduce_small(
        _sample_paths(sample_dir, artifact_root),
        seed=seed,
        max_epochs=max_epochs,
    )


@app.command("clean-sample")
def clean_sample_command(
    sample_dir: Path = SAMPLE_DIR_OPTION,
    artifact_root: Path = SAMPLE_ARTIFACT_ROOT_OPTION,
) -> None:
    """Remove generated sample artifacts only."""

    clean_sample(_sample_paths(sample_dir, artifact_root))


@app.command()
def data(
    sources_path: Path = SOURCES_PATH_OPTION,
    fetch_live_count: bool = FETCH_LIVE_COUNT_OPTION,
) -> None:
    """Inspect pinned Phase 1 data-source configuration."""

    sources = load_yaml(sources_path)["sources"]
    typer.echo("Pinned Phase 1 sources:")
    for name, spec in sources.items():
        version = spec.get("version", spec.get("status", "unknown"))
        license_name = spec.get("license", "n/a")
        typer.echo(f"- {name}: {version} ({license_name})")

    if fetch_live_count:
        ot = sources["open_targets"]
        count = fetch_oa_target_count(
            graphql_url=str(ot["graphql"]),
            disease_efo=str(ot["disease_efo"]),
        )
        typer.echo(f"Open Targets live OA associated-target count: {count}")


@app.command()
def graph(processed_dir: Path = PROCESSED_DIR_OPTION) -> None:
    """Validate and summarize processed graph tables if present."""

    nodes_path = processed_dir / "nodes.parquet"
    edges_dir = processed_dir / "edges"
    if not nodes_path.exists() or not edges_dir.exists():
        typer.echo("No processed graph found yet. Run source normalization first.")
        raise typer.Exit(code=0)

    nodes = read_table(nodes_path)
    edge_tables = {
        path.stem: read_table(path) for path in sorted(edges_dir.glob("*.parquet"))
    }
    assert_canonical_edge_types(edge_tables)
    stats = graph_statistics_table(nodes, edge_tables)
    typer.echo(stats.to_string(index=False))


@app.command()
def neo4j(
    processed_dir: Path = PROCESSED_DIR_OPTION,
    config_path: Path = CONFIG_PATH_OPTION,
) -> None:
    """Load processed graph tables into Neo4j idempotently."""

    settings = load_settings(config_path)
    nodes = read_table(processed_dir / "nodes.parquet")
    edge_tables = {
        path.stem: read_table(path)
        for path in sorted((processed_dir / "edges").glob("*.parquet"))
    }
    assert_canonical_edge_types(edge_tables)
    load_graph(
        nodes,
        edge_tables,
        Neo4jConfig(
            uri=settings.neo4j.uri,
            user=settings.neo4j.user,
            password=settings.neo4j.require_password(),
            database=settings.neo4j.database,
        ),
    )
    typer.echo("Neo4j load complete.")


@app.command()
def heterodata(
    processed_dir: Path = PROCESSED_DIR_OPTION,
    output_dir: Path = HETERODATA_DIR_OPTION,
    gene_feature_mode: str = typer.Option("structural"),
) -> None:
    """Export processed graph tables to a PyG HeteroData artifact."""

    data = build_heterodata(
        processed_dir=processed_dir,
        output_dir=output_dir,
        gene_feature_mode=gene_feature_mode,
    )
    node_types, edge_types = data.metadata()
    typer.echo(
        f"HeteroData export complete: {len(node_types)} node types, "
        f"{len(edge_types)} edge types."
    )


@app.command()
def splits(
    heterodata_path: Path = HETERODATA_PATH_OPTION,
    output_dir: Path = SPLITS_DIR_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    """Create and persist disjoint link-prediction supervision splits."""

    import torch

    data = torch.load(heterodata_path, weights_only=False)
    bundle = disjoint_random_link_split(data, seed=seed)
    save_splits(bundle, output_dir)
    typer.echo(f"Saved disjoint supervision splits to {output_dir}.")


@app.command()
def baselines() -> None:
    """List executable baseline outputs; use train-baselines to run them."""

    for name in (
        "random",
        "degree_popularity",
        "source_score_only",
        "logistic_regression",
        "gradient_boosted_trees",
        "matrix_factorization",
        "adjacency_svd",
        "node2vec",
        "transe",
        "distmult",
        "complex",
        "rotate",
        "hash_text",
        "feature_mlp",
    ):
        typer.echo(f"- {name}: artifacts/sample/metrics/baselines/{name}.json")


@app.command()
def train(
    splits_dir: Path = SPLITS_DIR_OPTION,
    heterodata_path: Path = HETERODATA_PATH_OPTION,
    output_dir: Path = MODEL_OUTPUT_DIR_OPTION,
    model_name: str = MODEL_NAME_OPTION,
    seed: int = SEED_OPTION,
    max_epochs: int = MAX_EPOCHS_OPTION,
) -> None:
    """Train a Phase 4 GNN model on disjoint split artifacts."""

    import torch

    bundle = load_splits(splits_dir)
    reference_data = torch.load(heterodata_path, weights_only=False)
    result = train_one_seed(
        bundle.train_data,
        bundle.val_data,
        bundle.test_data,
        reference_data,
        seed=seed,
        config=TrainingConfig(
            model_name=model_name,  # type: ignore[arg-type]
            max_epochs=max_epochs,
        ),
        output_dir=output_dir,
    )
    typer.echo(
        f"{model_name} seed {seed} complete: "
        f"AUPRC={result.metrics['AUPRC']:.4f}, "
        f"filtered_MRR={result.metrics['filtered_MRR']:.4f}."
    )


@app.command()
def ablate(
    reports_dir: Path = ABLATION_REPORTS_DIR_OPTION,
    output_dir: Path = ABLATION_TABLES_DIR_OPTION,
) -> None:
    """Assemble Phase 5 ablation markdown/LaTeX tables from saved reports."""

    paths = {
        name: reports_dir / f"{name}.json"
        for name in (
            "ablation1_nokg_vs_kg",
            "ablation2_kg_vs_kgtext",
            "ablation3_homo_rel_hetero",
            "ablation4_design_knobs",
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        typer.echo(
            "Missing ablation report JSON files: "
            + ", ".join(str(path) for path in missing)
        )
        raise typer.Exit(code=1)
    written = write_ablation_tables(
        read_report(paths["ablation1_nokg_vs_kg"]),
        read_report(paths["ablation2_kg_vs_kgtext"]),
        read_report(paths["ablation3_homo_rel_hetero"]),
        read_report(paths["ablation4_design_knobs"]),
        output_dir,
    )
    typer.echo(
        f"Wrote ablation tables to {written['markdown']} and {written['latex']}."
    )


@app.command()
def explain(
    checkpoint: Path = EXPLAIN_CHECKPOINT_OPTION,
    model_config: Path = EXPLAIN_MODEL_CONFIG_OPTION,
    graph: Path = EXPLAIN_GRAPH_OPTION,
    dataset_manifest: Path = EXPLAIN_DATASET_MANIFEST_OPTION,
    graph_manifest: Path = EXPLAIN_GRAPH_MANIFEST_OPTION,
    feature_manifest: Path = EXPLAIN_FEATURE_MANIFEST_OPTION,
    node_index_map: Path = EXPLAIN_NODE_INDEX_MAP_OPTION,
    split_metadata: Path = EXPLAIN_SPLIT_METADATA_OPTION,
    validation_metrics: Path = EXPLAIN_VALIDATION_METRICS_OPTION,
    run_manifest: Path = EXPLAIN_RUN_MANIFEST_OPTION,
    output_dir: Path = EXPLAIN_VALIDATED_OUTPUT_OPTION,
) -> None:
    """Explain predictions only from compatible, validated trained artifacts."""

    paths = ArtifactPaths(
        checkpoint=checkpoint,
        model_config=model_config,
        graph=graph,
        dataset_manifest=dataset_manifest,
        graph_manifest=graph_manifest,
        feature_manifest=feature_manifest,
        node_index_map=node_index_map,
        split_metadata=split_metadata,
        validation_metrics=validation_metrics,
        run_manifest=run_manifest,
    )
    try:
        artifacts = validate_and_load_artifacts(paths)
        result = run_validated_explanations(artifacts, output_dir)
    except (ArtifactValidationError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Explanation refused: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"Wrote {result['explanation_count']} validated explanations to {output_dir}."
    )


@app.command("smoke-train")
def smoke_train(
    tiny: bool = typer.Option(True, help="Use the built-in tiny synthetic graph."),
    output_dir: Path = SMOKE_OUTPUT_DIR_OPTION,
) -> None:
    """Run the CI smoke-train gate on a tiny HGT graph."""

    if not tiny:
        typer.echo("Only the built-in tiny smoke graph is supported by this gate.")
        raise typer.Exit(code=2)
    result = run_smoke_train(output_dir=output_dir)
    typer.echo(
        "Smoke-train complete: "
        f"AUPRC={result.metrics['AUPRC']:.4f}, "
        f"filtered_MRR={result.metrics['filtered_MRR']:.4f}."
    )


def main() -> None:
    """CLI entry point."""

    app()


if __name__ == "__main__":
    main()
