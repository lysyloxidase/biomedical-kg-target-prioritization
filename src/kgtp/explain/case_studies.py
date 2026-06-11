"""Publication-oriented interpretability case studies."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import networkx as nx
from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike
from kgtp.explain.attention import extract_hgt_attention_weights
from kgtp.explain.explainer import TargetExplainer
from kgtp.explain.metapaths import MetaPathExplanation, PathNode, rank_metapaths

KNOWN_TARGET_SYMBOLS: tuple[str, ...] = ("GDF5", "MMP13")
BIOLOGY_KEYWORDS: tuple[str, ...] = (
    "wnt",
    "beta-catenin",
    "tgf",
    "bmp",
    "cartilage",
    "extracellular matrix",
    "collagen",
    "matrix metalloproteinase",
)


@dataclass(frozen=True)
class PredictionCandidate:
    """A scored disease-gene prediction candidate."""

    disease_idx: int
    gene_idx: int
    score: float


@dataclass(frozen=True)
class CaseStudyResult:
    """Serializable interpretability deliverable for one prediction."""

    name: str
    disease_idx: int
    gene_idx: int
    disease_id: str
    gene_id: str
    gene_symbol: str
    model_score: float
    is_known_target: bool
    is_hypothesis: bool
    explanation_method: str
    biology_alignment: tuple[str, ...]
    top_paths: tuple[dict[str, object], ...]
    attention_summary: Mapping[str, float]
    explanatory_subgraph: Mapping[str, object]
    figure_paths: Mapping[str, str]
    narrative: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_known_target_case_study(
    explainer: TargetExplainer,
    data: HeteroData,
    *,
    disease_idx: int = 0,
    preferred_targets: Sequence[str] = KNOWN_TARGET_SYMBOLS,
    output_dir: PathLike = "reports/figures",
) -> CaseStudyResult:
    """Explain GDF5 or MMP13 when present in the graph."""

    gene_idx = select_known_target_index(data, preferred_targets)
    return _build_case_study(
        explainer,
        data,
        PredictionCandidate(disease_idx, gene_idx, math.nan),
        output_dir=output_dir,
        case_name=f"known_target_{_gene_symbol(data, gene_idx).lower()}",
        is_known_target=True,
        is_hypothesis=False,
    )


def build_novel_prediction_case_study(
    explainer: TargetExplainer,
    data: HeteroData,
    predictions: Sequence[PredictionCandidate | tuple[int, float]],
    train_positive_pairs: Iterable[tuple[int, int]],
    *,
    disease_idx: int = 0,
    output_dir: PathLike = "reports/figures",
) -> CaseStudyResult:
    """Explain the top prediction absent from training positives."""

    candidate = select_novel_prediction(
        predictions,
        train_positive_pairs,
        disease_idx=disease_idx,
        num_genes=int(data["gene"].num_nodes),
    )
    return _build_case_study(
        explainer,
        data,
        candidate,
        output_dir=output_dir,
        case_name=f"novel_hypothesis_{_gene_symbol(data, candidate.gene_idx).lower()}",
        is_known_target=False,
        is_hypothesis=True,
    )


def build_phase6_case_studies(
    explainer: TargetExplainer,
    data: HeteroData,
    predictions: Sequence[PredictionCandidate | tuple[int, float]],
    train_positive_pairs: Iterable[tuple[int, int]],
    *,
    disease_idx: int = 0,
    output_dir: PathLike = "reports/figures",
    min_predictions: int = 3,
) -> list[CaseStudyResult]:
    """Build known-target and hypothesis-framed case studies with figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    used_genes: set[int] = set()
    results: list[CaseStudyResult] = []

    for symbol in KNOWN_TARGET_SYMBOLS:
        gene_idx = find_gene_index(data, symbol)
        if gene_idx is None:
            continue
        used_genes.add(gene_idx)
        results.append(
            _build_case_study(
                explainer,
                data,
                PredictionCandidate(disease_idx, gene_idx, math.nan),
                output_dir=output,
                case_name=f"known_target_{symbol.lower()}",
                is_known_target=True,
                is_hypothesis=False,
            )
        )

    positives = set(train_positive_pairs)
    for candidate in _normalized_predictions(
        predictions,
        disease_idx=disease_idx,
        num_genes=int(data["gene"].num_nodes),
    ):
        if len(results) >= min_predictions:
            break
        if (
            candidate.gene_idx in used_genes
            or (
                candidate.disease_idx,
                candidate.gene_idx,
            )
            in positives
        ):
            continue
        used_genes.add(candidate.gene_idx)
        results.append(
            _build_case_study(
                explainer,
                data,
                candidate,
                output_dir=output,
                case_name=f"novel_hypothesis_{_gene_symbol(data, candidate.gene_idx).lower()}",
                is_known_target=False,
                is_hypothesis=True,
            )
        )

    while len(results) < min_predictions:
        fallback = select_novel_prediction(
            [],
            positives | {(result.disease_idx, result.gene_idx) for result in results},
            disease_idx=disease_idx,
            num_genes=int(data["gene"].num_nodes),
        )
        results.append(
            _build_case_study(
                explainer,
                data,
                fallback,
                output_dir=output,
                case_name=f"novel_hypothesis_{_gene_symbol(data, fallback.gene_idx).lower()}",
                is_known_target=False,
                is_hypothesis=True,
            )
        )

    payload_path = output / "case_studies.json"
    payload_path.write_text(
        json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_case_study_narrative(results, output / "case_study_narrative.md")
    return results


def select_known_target_index(
    data: HeteroData,
    preferred_targets: Sequence[str] = KNOWN_TARGET_SYMBOLS,
) -> int:
    """Return the first preferred known target present in gene symbols or IDs."""

    for symbol in preferred_targets:
        gene_idx = find_gene_index(data, symbol)
        if gene_idx is not None:
            return gene_idx
    msg = (
        f"None of the preferred known targets were present: {tuple(preferred_targets)}"
    )
    raise ValueError(msg)


def select_novel_prediction(
    predictions: Sequence[PredictionCandidate | tuple[int, float]],
    train_positive_pairs: Iterable[tuple[int, int]],
    *,
    disease_idx: int,
    num_genes: int,
) -> PredictionCandidate:
    """Select the top-ranked candidate absent from training positives."""

    positives = set(train_positive_pairs)
    for candidate in _normalized_predictions(
        predictions,
        disease_idx=disease_idx,
        num_genes=num_genes,
    ):
        if (candidate.disease_idx, candidate.gene_idx) not in positives:
            return candidate

    for gene_idx in range(num_genes):
        if (disease_idx, gene_idx) not in positives:
            return PredictionCandidate(disease_idx, gene_idx, math.nan)
    msg = "No novel disease-gene candidate is available"
    raise ValueError(msg)


def find_gene_index(data: HeteroData, symbol_or_id: str) -> int | None:
    """Find a gene row by symbol, label, or node ID."""

    needle = symbol_or_id.lower()
    for index, value in enumerate(_node_ids(data, "gene")):
        if value.lower() == needle:
            return index
    for attr in ("symbol", "label", "name", "node_label"):
        if not hasattr(data["gene"], attr):
            continue
        values = [str(value) for value in getattr(data["gene"], attr)]
        for index, value in enumerate(values):
            if value.lower() == needle:
                return index
    return None


def save_explanatory_subgraph_figure(
    data: HeteroData,
    case: CaseStudyResult,
    paths: Sequence[MetaPathExplanation],
    output_path: PathLike,
) -> dict[str, str]:
    """Save a deterministic NetworkX/matplotlib explanatory subgraph figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = nx.DiGraph()
    for path in paths:
        for node in path.nodes:
            _add_figure_node(graph, node)
        for left, right, edge_type in zip(
            path.nodes[:-1],
            path.nodes[1:],
            path.edge_types,
            strict=True,
        ):
            graph.add_edge(_node_key(left), _node_key(right), label=edge_type[1])

    if graph.number_of_nodes() == 0:
        disease_node = PathNode(
            "disease",
            case.disease_idx,
            case.disease_id,
            _node_label(data, "disease", case.disease_idx),
        )
        gene_node = PathNode("gene", case.gene_idx, case.gene_id, case.gene_symbol)
        _add_figure_node(graph, disease_node)
        _add_figure_node(graph, gene_node)
        graph.add_edge(_node_key(disease_node), _node_key(gene_node), label="predicted")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=13, k=1.4)
    colors = [
        _node_color(str(attrs["node_type"])) for _, attrs in graph.nodes(data=True)
    ]
    labels = {node: str(attrs["label"]) for node, attrs in graph.nodes(data=True)}
    edge_labels = nx.get_edge_attributes(graph, "label")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=colors,
        node_size=1300,
        linewidths=1.2,
        edgecolors="#2f3a44",
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.5,
        edge_color="#6b7280",
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=8, font_weight="bold", ax=axis
    )
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=7, ax=axis
    )
    axis.set_title(case.name.replace("_", " "), fontsize=11, fontweight="bold")
    axis.axis("off")
    figure.savefig(output, dpi=220)
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path)
    plt.close(figure)
    return {"png": str(output), "pdf": str(pdf_path)}


def write_case_study_narrative(
    results: Sequence[CaseStudyResult],
    path: PathLike,
) -> Path:
    """Write an honest Phase 6 narrative markdown file."""

    lines = [
        "# Phase 6 Interpretability Case Studies",
        "",
        "These explanations are qualitative model rationales under sparse supervision. "
        "Novel predictions are computational hypotheses, not validated targets.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.name}",
                "",
                result.narrative,
                "",
                f"- Gene: {result.gene_symbol} ({result.gene_id})",
                f"- Known target: {result.is_known_target}",
                f"- Hypothesis flagged: {result.is_hypothesis}",
                f"- Biology alignment: {', '.join(result.biology_alignment) or 'not detected'}",
                f"- Figure: {result.figure_paths.get('png', '')}",
                "",
            ]
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _build_case_study(
    explainer: TargetExplainer,
    data: HeteroData,
    candidate: PredictionCandidate,
    *,
    output_dir: PathLike,
    case_name: str,
    is_known_target: bool,
    is_hypothesis: bool,
) -> CaseStudyResult:
    explanation = explainer.explain_link(
        candidate.disease_idx, candidate.gene_idx, data
    )
    subgraph = explainer.explanatory_subgraph(explanation)
    paths = rank_metapaths(data, candidate.disease_idx, candidate.gene_idx, max_paths=8)
    attention = extract_hgt_attention_weights(
        explainer.model,
        data,
        candidate.disease_idx,
        candidate.gene_idx,
        edge_type=explainer.edge_type,
        top_k=20,
    )
    biology_alignment = tuple(_biology_alignment(paths))
    gene_symbol = _gene_symbol(data, candidate.gene_idx)
    model_score = (
        candidate.score
        if not math.isnan(candidate.score)
        else float(getattr(explanation, "score", math.nan))
    )
    result = CaseStudyResult(
        name=case_name,
        disease_idx=candidate.disease_idx,
        gene_idx=candidate.gene_idx,
        disease_id=_node_ids(data, "disease")[candidate.disease_idx],
        gene_id=_node_ids(data, "gene")[candidate.gene_idx],
        gene_symbol=gene_symbol,
        model_score=model_score,
        is_known_target=is_known_target,
        is_hypothesis=is_hypothesis,
        explanation_method=str(getattr(explanation, "method", "unknown")),
        biology_alignment=biology_alignment,
        top_paths=tuple(path.to_dict() for path in paths),
        attention_summary=cast(Mapping[str, float], attention["meta_relation_summary"]),
        explanatory_subgraph=cast(Mapping[str, object], subgraph),
        figure_paths={},
        narrative=_case_narrative(
            gene_symbol,
            is_known_target=is_known_target,
            is_hypothesis=is_hypothesis,
            biology_alignment=biology_alignment,
            paths=paths,
        ),
    )
    figure_paths = save_explanatory_subgraph_figure(
        data,
        result,
        paths,
        Path(output_dir) / f"{case_name}.png",
    )
    return CaseStudyResult(
        name=result.name,
        disease_idx=result.disease_idx,
        gene_idx=result.gene_idx,
        disease_id=result.disease_id,
        gene_id=result.gene_id,
        gene_symbol=result.gene_symbol,
        model_score=result.model_score,
        is_known_target=result.is_known_target,
        is_hypothesis=result.is_hypothesis,
        explanation_method=result.explanation_method,
        biology_alignment=result.biology_alignment,
        top_paths=result.top_paths,
        attention_summary=result.attention_summary,
        explanatory_subgraph=result.explanatory_subgraph,
        figure_paths=figure_paths,
        narrative=result.narrative,
    )


def _case_narrative(
    gene_symbol: str,
    *,
    is_known_target: bool,
    is_hypothesis: bool,
    biology_alignment: Sequence[str],
    paths: Sequence[MetaPathExplanation],
) -> str:
    if is_known_target:
        prefix = (
            f"{gene_symbol} is treated as a known-target sanity check. The rationale "
            "should recover OA-relevant neighborhood evidence rather than be read as "
            "a new discovery."
        )
    elif is_hypothesis:
        prefix = (
            f"{gene_symbol} is a computational hypothesis generated from the model "
            "ranking. It is not a validated osteoarthritis target and requires "
            "independent biological validation."
        )
    else:
        prefix = f"{gene_symbol} is explained as a model-ranked candidate."

    path_text = (
        f" Top-ranked paths include {len(paths)} shared-pathway or PPI rationales."
        if paths
        else " No compact shared-pathway or PPI rationale was found in the current graph."
    )
    biology_text = (
        " Detected OA biology terms: " + ", ".join(biology_alignment) + "."
        if biology_alignment
        else " No Wnt/TGF-beta/BMP/PPI biology keyword was detected in this small graph."
    )
    caveat = (
        " Heterogeneous explainers remain an immature interface, so these masks and "
        "attention values are qualitative support for inspection, not causal proof."
    )
    return prefix + path_text + biology_text + caveat


def _biology_alignment(paths: Sequence[MetaPathExplanation]) -> list[str]:
    text = " ".join(
        " ".join(node.label for node in path.nodes) + " " + path.evidence
        for path in paths
    ).lower()
    hits = [keyword for keyword in BIOLOGY_KEYWORDS if keyword in text]
    return sorted(set(hits))


def _normalized_predictions(
    predictions: Sequence[PredictionCandidate | tuple[int, float]],
    *,
    disease_idx: int,
    num_genes: int,
) -> list[PredictionCandidate]:
    normalized: list[PredictionCandidate] = []
    for item in predictions:
        if isinstance(item, PredictionCandidate):
            normalized.append(item)
        else:
            gene_idx, score = item
            normalized.append(
                PredictionCandidate(disease_idx, int(gene_idx), float(score))
            )
    normalized = [
        candidate for candidate in normalized if 0 <= candidate.gene_idx < num_genes
    ]
    normalized.sort(key=lambda candidate: candidate.score, reverse=True)
    return normalized


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]


def _gene_symbol(data: HeteroData, gene_idx: int) -> str:
    for attr in ("symbol", "label", "name", "node_label"):
        if hasattr(data["gene"], attr):
            values = [str(value) for value in getattr(data["gene"], attr)]
            return values[gene_idx]
    return _node_ids(data, "gene")[gene_idx]


def _node_label(data: HeteroData, node_type: str, index: int) -> str:
    for attr in ("symbol", "label", "name", "node_label"):
        if hasattr(data[node_type], attr):
            values = [str(value) for value in getattr(data[node_type], attr)]
            return values[index]
    return _node_ids(data, node_type)[index]


def _node_key(node: PathNode) -> str:
    return f"{node.node_type}:{node.index}"


def _add_figure_node(graph: nx.DiGraph, node: PathNode) -> None:
    graph.add_node(
        _node_key(node),
        node_type=node.node_type,
        label=node.label,
        node_id=node.node_id,
    )


def _node_color(node_type: str) -> str:
    return {
        "disease": "#f2b8a2",
        "gene": "#9ecae1",
        "pathway": "#a1d99b",
        "drug": "#c7b9ff",
        "go_term": "#fdd878",
    }.get(node_type, "#d1d5db")
