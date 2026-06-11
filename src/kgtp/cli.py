"""Command-line interface for the benchmark pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from kgtp.ablation.common import read_report
from kgtp.ablation.tables import write_ablation_tables
from kgtp.config import load_settings, load_yaml
from kgtp.data.build_graph import assert_canonical_edge_types
from kgtp.data.common import read_table
from kgtp.data.opentargets import fetch_oa_target_count
from kgtp.explain.case_studies import PredictionCandidate, build_phase6_case_studies
from kgtp.explain.explainer import DISEASE_GENE_EDGE, TargetExplainer
from kgtp.hetero.build_heterodata import build_heterodata
from kgtp.hetero.splits import leakage_free_random_link_split, load_splits, save_splits
from kgtp.kg.neo4j_loader import Neo4jConfig, load_graph
from kgtp.kg.statistics import graph_statistics_table
from kgtp.models.train import TrainingConfig, build_model, train_one_seed
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
SMOKE_OUTPUT_DIR_OPTION = typer.Option(Path("reports/smoke_train"))


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
            password=settings.neo4j.password,
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
    """Create and persist leakage-free link-prediction splits."""

    import torch

    data = torch.load(heterodata_path, weights_only=False)
    bundle = leakage_free_random_link_split(data, seed=seed)
    save_splits(bundle, output_dir)
    typer.echo(f"Saved leakage-free splits to {output_dir}.")


@app.command()
def baselines() -> None:
    """List Phase 3 non-graph baselines and result output convention."""

    for name in (
        "popularity",
        "logistic_regression",
        "matrix_factorization",
        "text_embeddings",
        "node2vec",
        "centrality",
        "kge",
    ):
        typer.echo(f"- {name}: reports/results_{name}.json")


@app.command()
def train(
    splits_dir: Path = SPLITS_DIR_OPTION,
    heterodata_path: Path = HETERODATA_PATH_OPTION,
    output_dir: Path = MODEL_OUTPUT_DIR_OPTION,
    model_name: str = MODEL_NAME_OPTION,
    seed: int = SEED_OPTION,
    max_epochs: int = MAX_EPOCHS_OPTION,
) -> None:
    """Train a Phase 4 GNN model on leakage-free split artifacts."""

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
    heterodata_path: Path = HETERODATA_PATH_OPTION,
    output_dir: Path = EXPLAIN_OUTPUT_DIR_OPTION,
) -> None:
    """Build Phase 6 interpretability case-study figures from HeteroData."""

    import torch

    if not heterodata_path.exists():
        typer.echo(f"No HeteroData artifact found at {heterodata_path}.")
        raise typer.Exit(code=0)
    data = torch.load(heterodata_path, weights_only=False)
    model = build_model(
        data.metadata(),
        TrainingConfig(
            model_name="hgt",
            hidden_channels=16,
            num_heads=4,
            num_layers=1,
            edge_types=(DISEASE_GENE_EDGE,),
            negatives_per_positive=16,
        ),
    )
    gene_count = int(data["gene"].num_nodes)
    predictions = [
        PredictionCandidate(0, gene_idx, float(gene_count - gene_idx))
        for gene_idx in range(gene_count)
    ]
    train_positive_pairs: set[tuple[int, int]] = set()
    if DISEASE_GENE_EDGE in data.edge_types:
        for source, target in data[DISEASE_GENE_EDGE].edge_index.t().tolist():
            train_positive_pairs.add((int(source), int(target)))
    explainer = TargetExplainer(
        model,
        data,
        edge_type=DISEASE_GENE_EDGE,
        integration_steps=8,
    )
    results = build_phase6_case_studies(
        explainer,
        data,
        predictions,
        train_positive_pairs,
        output_dir=output_dir,
    )
    typer.echo(f"Wrote {len(results)} explanation case studies to {output_dir}.")


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
