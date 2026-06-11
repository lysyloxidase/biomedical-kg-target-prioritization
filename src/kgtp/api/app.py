"""Optional FastAPI target-ranking endpoint."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from torch_geometric.data import HeteroData

from kgtp.explain.explainer import DISEASE_GENE_EDGE, TargetExplainer
from kgtp.models.train import TrainingConfig, build_model
from kgtp.smoke import tiny_heterodata

EdgeType = tuple[str, str, str]

app = FastAPI(title="biomedical-kg-target-prioritization", version="1.0")


@dataclass
class PredictionService:
    """Loaded model, graph, and explainer state for the API."""

    data: HeteroData
    model: torch.nn.Module
    explainer: TargetExplainer
    artifact_source: str
    model_source: str
    model_warning: str | None = None


@app.get("/", response_class=HTMLResponse)
async def project_dashboard() -> HTMLResponse:
    """Return the browser-visible project dashboard with graph visualization."""

    return HTMLResponse(PROJECT_DASHBOARD_HTML)


@app.get("/health")
async def health() -> dict[str, object]:
    """Return API health and artifact-loading status."""

    service = get_service()
    return {
        "status": "ok",
        "artifact_source": service.artifact_source,
        "model_source": service.model_source,
        "model_warning": service.model_warning,
        "disclaimer": "Computational hypothesis generation only; not validated target discovery or clinical advice.",
    }


@app.get("/graph-stats")
async def graph_stats() -> dict[str, object]:
    """Return graph-statistics table derived from the loaded ``HeteroData``."""

    service = get_service()
    stats = graph_stats_from_heterodata(service.data)
    return stats


@app.get("/graph-data")
async def graph_data() -> dict[str, object]:
    """Return browser-ready graph nodes and edges."""

    service = get_service()
    return graph_data_from_heterodata(service.data)


@app.get("/predict")
async def predict(
    disease: str = "EFO_0004616",
    top_k: int = Query(20, ge=1, le=100),
) -> dict[str, object]:
    """Rank candidate target genes and attach compact explanation subgraphs."""

    service = get_service()
    disease_idx = _find_node_index(service.data, "disease", disease)
    if disease_idx is None:
        raise HTTPException(status_code=404, detail=f"Disease not found: {disease}")

    ranked = rank_targets(service, disease_idx, top_k=top_k)
    return {
        "disease": disease,
        "primary_metric": "AUPRC",
        "caveat": "Ranked genes are computational hypotheses and require independent validation.",
        "results": ranked,
    }


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    """Load artifacts once, falling back to a deterministic tiny graph."""

    torch.manual_seed(13)
    heterodata_path = Path(
        os.environ.get(
            "KGTP_HETERODATA_PATH", "data/processed/heterodata/heterodata.pt"
        )
    )
    model_path = Path(
        os.environ.get("KGTP_MODEL_PATH", "reports/models/hgt_seed13/model.pt")
    )
    hidden_channels = int(os.environ.get("KGTP_HIDDEN_CHANNELS", "16"))
    num_heads = int(os.environ.get("KGTP_NUM_HEADS", "4"))
    num_layers = int(os.environ.get("KGTP_NUM_LAYERS", "1"))

    if heterodata_path.exists():
        data = cast(HeteroData, torch.load(heterodata_path, weights_only=False))
        artifact_source = str(heterodata_path)
    else:
        data = tiny_heterodata()
        artifact_source = "tiny-smoke-graph"

    model = build_model(
        data.metadata(),
        TrainingConfig(
            model_name="hgt",
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            edge_types=(DISEASE_GENE_EDGE,),
            negatives_per_positive=16,
        ),
    )
    _initialize_lazy_modules(model, data)
    model_source = "untrained-smoke-model"
    model_warning: str | None = None
    if model_path.exists():
        try:
            state = torch.load(model_path, weights_only=False, map_location="cpu")
            model.load_state_dict(state)
            model_source = str(model_path)
        except Exception as exc:
            model_warning = f"Could not load model state from {model_path}: {type(exc).__name__}: {exc}"

    explainer = TargetExplainer(
        model,
        data,
        edge_type=DISEASE_GENE_EDGE,
        integration_steps=1,
        use_pyg_captum=False,
    )
    return PredictionService(
        data=data,
        model=model,
        explainer=explainer,
        artifact_source=artifact_source,
        model_source=model_source,
        model_warning=model_warning,
    )


def rank_targets(
    service: PredictionService,
    disease_idx: int,
    *,
    top_k: int,
) -> list[dict[str, object]]:
    """Score all genes for one disease and return top-ranked predictions."""

    data = service.data
    model = service.model
    model.eval()
    with torch.no_grad():
        z_dict = cast(Any, model).encode(data.x_dict, _edge_index_dict(data))
        gene_count = int(data["gene"].num_nodes)
        edge_label_index = torch.stack(
            [
                torch.full((gene_count,), disease_idx, dtype=torch.long),
                torch.arange(gene_count, dtype=torch.long),
            ]
        )
        scores = cast(
            torch.Tensor,
            cast(Any, model).decode(z_dict, DISEASE_GENE_EDGE, edge_label_index),
        )

    known_pairs = _known_disease_gene_pairs(data)
    gene_ids = _node_ids(data, "gene")
    symbols = _node_labels(data, "gene")
    order = torch.argsort(scores, descending=True)[:top_k].tolist()
    rows: list[dict[str, object]] = []
    for rank, gene_idx in enumerate(order, start=1):
        score = float(scores[int(gene_idx)].item())
        explanation = service.explainer.explain_link(disease_idx, int(gene_idx), data)
        subgraph = service.explainer.explanatory_subgraph(
            explanation,
            top_k_nodes=5,
            top_k_edges=8,
        )
        rows.append(
            {
                "rank": rank,
                "gene_id": gene_ids[int(gene_idx)],
                "gene_symbol": symbols[int(gene_idx)],
                "score": score,
                "known_training_positive": (disease_idx, int(gene_idx)) in known_pairs,
                "hypothesis": (disease_idx, int(gene_idx)) not in known_pairs,
                "explanation_subgraph": subgraph,
            }
        )
    return rows


def graph_data_from_heterodata(data: HeteroData) -> dict[str, object]:
    """Convert ``HeteroData`` to a compact graph payload for the web UI."""

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    degrees: dict[str, int] = {}
    for node_type in data.node_types:
        ids = _node_ids(data, node_type)
        labels = _node_labels(data, node_type)
        for index, node_id in enumerate(ids):
            key = _graph_node_key(node_type, index)
            degrees[key] = 0
            nodes.append(
                {
                    "id": key,
                    "node_type": node_type,
                    "index": index,
                    "node_id": node_id,
                    "label": labels[index],
                }
            )

    for edge_type in data.edge_types:
        if not hasattr(data[edge_type], "edge_index"):
            continue
        source_type, relation, target_type = cast(EdgeType, edge_type)
        edge_index = data[edge_type].edge_index
        for position, (source, target) in enumerate(edge_index.t().tolist()):
            source_key = _graph_node_key(source_type, int(source))
            target_key = _graph_node_key(target_type, int(target))
            degrees[source_key] = degrees.get(source_key, 0) + 1
            degrees[target_key] = degrees.get(target_key, 0) + 1
            edges.append(
                {
                    "id": f"{'__'.join(cast(EdgeType, edge_type))}:{position}",
                    "source": source_key,
                    "target": target_key,
                    "edge_type": list(cast(EdgeType, edge_type)),
                    "relation": relation,
                }
            )

    for node in nodes:
        node["degree"] = degrees[str(node["id"])]
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_types": data.node_types,
            "edge_types": [
                list(cast(EdgeType, edge_type)) for edge_type in data.edge_types
            ],
        },
    }


def graph_stats_from_heterodata(data: HeteroData) -> dict[str, object]:
    """Compute a compact graph-statistics table directly from HeteroData."""

    node_counts = {
        node_type: int(data[node_type].num_nodes) for node_type in data.node_types
    }
    edge_counts = {
        "__".join(cast(EdgeType, edge_type)): int(data[edge_type].edge_index.size(1))
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }
    node_total = sum(node_counts.values())
    edge_total = sum(edge_counts.values())
    density = edge_total / (node_total * (node_total - 1)) if node_total > 1 else 0.0
    mean_degree = (2 * edge_total / node_total) if node_total else 0.0
    positive_links = edge_counts.get("__".join(DISEASE_GENE_EDGE), 0)
    table = [
        {"metric": "node_count", "name": name, "value": count}
        for name, count in sorted(node_counts.items())
    ]
    table.extend(
        {"metric": "edge_count", "name": name, "value": count}
        for name, count in sorted(edge_counts.items())
    )
    table.extend(
        [
            {"metric": "density", "name": "global", "value": density},
            {"metric": "mean_degree", "name": "global", "value": mean_degree},
            {
                "metric": "positive_disease_gene_links",
                "name": "disease__associated_with__gene",
                "value": positive_links,
            },
        ]
    )
    return {
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "density": density,
        "mean_degree": mean_degree,
        "positive_disease_gene_links": positive_links,
        "table": table,
    }


def _initialize_lazy_modules(model: torch.nn.Module, data: HeteroData) -> None:
    with torch.no_grad():
        try:
            cast(Any, model).encode(data.x_dict, _edge_index_dict(data))
        except Exception:
            return


def _edge_index_dict(data: HeteroData) -> dict[EdgeType, torch.Tensor]:
    return {
        cast(EdgeType, edge_type): data[edge_type].edge_index
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_index")
    }


def _known_disease_gene_pairs(data: HeteroData) -> set[tuple[int, int]]:
    if DISEASE_GENE_EDGE not in data.edge_types:
        return set()
    return {
        (int(source), int(target))
        for source, target in data[DISEASE_GENE_EDGE].edge_index.t().tolist()
    }


def _find_node_index(data: HeteroData, node_type: str, node_id: str) -> int | None:
    lowered = node_id.lower()
    for index, value in enumerate(_node_ids(data, node_type)):
        if value.lower() == lowered:
            return index
    for index, value in enumerate(_node_labels(data, node_type)):
        if value.lower() == lowered:
            return index
    return None


def _node_ids(data: HeteroData, node_type: str) -> list[str]:
    if hasattr(data[node_type], "node_id"):
        return [str(value) for value in data[node_type].node_id]
    return [str(index) for index in range(int(data[node_type].num_nodes))]


def _node_labels(data: HeteroData, node_type: str) -> list[str]:
    for attr in ("symbol", "label", "name", "node_label"):
        if hasattr(data[node_type], attr):
            return [str(value) for value in getattr(data[node_type], attr)]
    ids = _node_ids(data, node_type)
    return (
        ids if ids else [str(index) for index in range(int(data[node_type].num_nodes))]
    )


def _graph_node_key(node_type: str, index: int) -> str:
    return f"{node_type}:{index}"


def _safe_float(value: object) -> float:
    try:
        return float(cast(float, value))
    except (TypeError, ValueError):
        return math.nan


PROJECT_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>biomedical-kg-target-prioritization</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --surface: #f6f8fb;
      --panel: #ffffff;
      --disease: #e66a4e;
      --gene: #2374ab;
      --drug: #7a5ea8;
      --pathway: #2d8f67;
      --go: #b88720;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--surface);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 760;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #16a34a;
      box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.12);
    }
    main {
      height: calc(100vh - 64px);
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 390px;
      grid-template-rows: 1fr 210px;
      gap: 1px;
      background: var(--line);
    }
    section, aside {
      background: var(--panel);
      min-width: 0;
      min-height: 0;
    }
    #graph-pane {
      position: relative;
      overflow: hidden;
    }
    #graph {
      width: 100%;
      height: 100%;
      display: block;
      background: #fbfcfe;
    }
    .toolbar {
      position: absolute;
      left: 16px;
      top: 16px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      max-width: calc(100% - 32px);
      padding: 8px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(6px);
    }
    button {
      height: 32px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      padding: 0 10px;
      cursor: pointer;
    }
    button.active {
      border-color: #17202a;
      background: #17202a;
      color: #fff;
    }
    #side {
      display: grid;
      grid-template-rows: 190px 1fr;
    }
    .panel {
      padding: 18px;
      border-bottom: 1px solid var(--line);
      overflow: auto;
    }
    .panel:last-child { border-bottom: 0; }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
      font-weight: 760;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      padding: 10px;
      min-height: 62px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.1;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 9px 6px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 7px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    #bottom {
      grid-column: 1 / 3;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 1px;
      background: var(--line);
    }
    #bottom .panel {
      background: var(--panel);
      border-bottom: 0;
    }
    .node-label {
      font-size: 11px;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 4px;
      stroke-linejoin: round;
      fill: #17202a;
      pointer-events: none;
    }
    .node circle {
      stroke: #ffffff;
      stroke-width: 2px;
      cursor: pointer;
    }
    .node.selected circle {
      stroke: #111827;
      stroke-width: 3px;
    }
    .edge {
      stroke: #a8b2c1;
      stroke-opacity: 0.58;
      stroke-width: 1.3px;
    }
    .edge.highlight {
      stroke: #111827;
      stroke-opacity: 0.86;
      stroke-width: 2.5px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 24px;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .kv {
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 8px;
      font-size: 13px;
      margin: 8px 0;
    }
    .kv span:first-child { color: var(--muted); }
    .hypothesis { color: #8a4b00; font-weight: 700; }
    .known { color: #0f766e; font-weight: 700; }
    @media (max-width: 960px) {
      main {
        height: auto;
        min-height: calc(100vh - 64px);
        grid-template-columns: 1fr;
        grid-template-rows: 58vh auto auto;
      }
      #bottom {
        grid-column: 1;
        grid-template-columns: 1fr;
      }
      #side { grid-template-rows: auto auto; }
      header { align-items: flex-start; height: auto; padding: 14px 16px; gap: 10px; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>biomedical-kg-target-prioritization</h1>
    <div class="status"><span class="dot"></span><span id="source">loading graph</span></div>
  </header>
  <main>
    <section id="graph-pane">
      <svg id="graph" role="img" aria-label="Heterogeneous biomedical knowledge graph"></svg>
      <div class="toolbar" id="filters"></div>
    </section>
    <aside id="side">
      <div class="panel">
        <h2>Graph Statistics</h2>
        <div class="stats-grid" id="stats"></div>
      </div>
      <div class="panel">
        <h2>Selected Node</h2>
        <div id="selection"></div>
      </div>
    </aside>
    <section id="bottom">
      <div class="panel">
        <h2>Target Ranking</h2>
        <table>
          <thead><tr><th>#</th><th>Gene</th><th>Score</th><th>Flag</th></tr></thead>
          <tbody id="ranking"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Explanation Subgraph</h2>
        <div id="explanation"></div>
      </div>
      <div class="panel">
        <h2>Project Scope</h2>
        <div class="legend" id="legend"></div>
        <div class="kv"><span>Primary</span><strong>AUPRC</strong></div>
        <div class="kv"><span>Protocol</span><strong>filtered OGB-style link prediction</strong></div>
        <div class="kv"><span>Models</span><strong>HGT, GraphSAGE, R-GCN, seven baselines</strong></div>
        <div class="kv"><span>Caveat</span><strong>computational hypotheses, not validated targets</strong></div>
      </div>
    </section>
  </main>
  <script>
    const colors = {
      disease: '#e66a4e',
      gene: '#2374ab',
      drug: '#7a5ea8',
      pathway: '#2d8f67',
      go_term: '#b88720'
    };
    const typeOrder = ['disease', 'gene', 'pathway', 'drug', 'go_term'];
    const state = { graph: null, visible: new Set(typeOrder), selected: null, ranking: [] };

    async function boot() {
      const [health, graph, stats, prediction] = await Promise.all([
        fetch('/health').then(r => r.json()),
        fetch('/graph-data').then(r => r.json()),
        fetch('/graph-stats').then(r => r.json()),
        fetch('/predict?disease=EFO_0004616&top_k=8').then(r => r.json())
      ]);
      state.graph = graph;
      state.ranking = prediction.results || [];
      document.getElementById('source').textContent = `${health.artifact_source} / ${health.model_source}`;
      renderFilters(graph.metadata.node_types);
      renderLegend(graph.metadata.node_types);
      renderStats(stats);
      renderRanking(state.ranking);
      renderExplanation(state.ranking[0]);
      renderSelection(null);
      renderGraph();
    }

    function renderFilters(types) {
      const box = document.getElementById('filters');
      box.innerHTML = '';
      types.forEach(type => {
        const button = document.createElement('button');
        button.textContent = type.replace('_', ' ');
        button.className = 'active';
        button.onclick = () => {
          if (state.visible.has(type)) state.visible.delete(type);
          else state.visible.add(type);
          button.classList.toggle('active', state.visible.has(type));
          renderGraph();
        };
        box.appendChild(button);
      });
    }

    function renderLegend(types) {
      const legend = document.getElementById('legend');
      legend.innerHTML = '';
      types.forEach(type => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="swatch" style="background:${colors[type] || '#64748b'}"></span>${type}`;
        legend.appendChild(item);
      });
    }

    function renderStats(stats) {
      const statsBox = document.getElementById('stats');
      const cells = [
        ['Nodes', Object.values(stats.node_counts).reduce((a, b) => a + b, 0)],
        ['Edges', Object.values(stats.edge_counts).reduce((a, b) => a + b, 0)],
        ['Disease-gene+', stats.positive_disease_gene_links],
        ['Mean degree', Number(stats.mean_degree).toFixed(2)]
      ];
      statsBox.innerHTML = cells.map(([label, value]) =>
        `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`
      ).join('');
    }

    function renderRanking(rows) {
      const tbody = document.getElementById('ranking');
      tbody.innerHTML = rows.map(row => {
        const flag = row.hypothesis
          ? '<span class="hypothesis">hypothesis</span>'
          : '<span class="known">known</span>';
        return `<tr data-gene="${row.gene_id}">
          <td>${row.rank}</td><td><strong>${row.gene_symbol}</strong><br><span class="tag">${row.gene_id}</span></td>
          <td>${Number(row.score).toFixed(4)}</td><td>${flag}</td>
        </tr>`;
      }).join('');
      [...tbody.querySelectorAll('tr')].forEach((tr, index) => {
        tr.onclick = () => renderExplanation(rows[index]);
      });
    }

    function renderExplanation(row) {
      const box = document.getElementById('explanation');
      if (!row) {
        box.innerHTML = '<span class="tag">waiting for ranking</span>';
        return;
      }
      const sub = row.explanation_subgraph || {};
      const nodes = (sub.nodes || []).slice(0, 5).map(node =>
        `<div class="kv"><span>${node.node_type}</span><strong>${node.node_id}</strong></div>`
      ).join('');
      const edges = (sub.edges || []).slice(0, 4).map(edge =>
        `<div class="kv"><span>${edge.edge_type[1]}</span><strong>${edge.source} -> ${edge.target}</strong></div>`
      ).join('');
      box.innerHTML = `
        <div class="kv"><span>Prediction</span><strong>${row.gene_symbol} (${row.gene_id})</strong></div>
        <div class="kv"><span>Method</span><strong>${sub.method || 'n/a'}</strong></div>
        <div class="kv"><span>Flag</span><strong>${row.hypothesis ? 'computational hypothesis' : 'known positive'}</strong></div>
        ${nodes}
        ${edges}
      `;
    }

    function renderSelection(node) {
      const box = document.getElementById('selection');
      if (!node) {
        box.innerHTML = '<span class="tag">none</span>';
        return;
      }
      box.innerHTML = `
        <div class="kv"><span>Label</span><strong>${node.label}</strong></div>
        <div class="kv"><span>ID</span><strong>${node.node_id}</strong></div>
        <div class="kv"><span>Type</span><strong>${node.node_type}</strong></div>
        <div class="kv"><span>Degree</span><strong>${node.degree}</strong></div>
      `;
    }

    function renderGraph() {
      const svg = document.getElementById('graph');
      const width = svg.clientWidth || 900;
      const height = svg.clientHeight || 620;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.innerHTML = '';
      const nodes = state.graph.nodes.filter(node => state.visible.has(node.node_type));
      const nodeIds = new Set(nodes.map(node => node.id));
      const edges = state.graph.edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target));
      const nodeMap = new Map(nodes.map(node => [node.id, {...node}]));
      layout(nodes, width, height);

      const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      const nodeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      svg.appendChild(edgeLayer);
      svg.appendChild(nodeLayer);

      edges.forEach(edge => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) return;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', source.x);
        line.setAttribute('y1', source.y);
        line.setAttribute('x2', target.x);
        line.setAttribute('y2', target.y);
        line.setAttribute('class', `edge ${state.selected && (edge.source === state.selected || edge.target === state.selected) ? 'highlight' : ''}`);
        edgeLayer.appendChild(line);
      });

      nodes.forEach(node => {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', `node ${state.selected === node.id ? 'selected' : ''}`);
        group.setAttribute('transform', `translate(${node.x}, ${node.y})`);
        group.onclick = () => {
          state.selected = node.id;
          renderSelection(node);
          renderGraph();
        };
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('r', Math.max(9, Math.min(18, 8 + node.degree * 0.9)));
        circle.setAttribute('fill', colors[node.node_type] || '#64748b');
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'node-label');
        label.setAttribute('y', -16);
        label.setAttribute('text-anchor', 'middle');
        label.textContent = shortLabel(node.label);
        group.appendChild(circle);
        group.appendChild(label);
        nodeLayer.appendChild(group);
      });
    }

    function layout(nodes, width, height) {
      const groups = new Map();
      nodes.forEach(node => {
        if (!groups.has(node.node_type)) groups.set(node.node_type, []);
        groups.get(node.node_type).push(node);
      });
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.34;
      typeOrder.forEach((type, typeIndex) => {
        const group = groups.get(type) || [];
        const angle = (Math.PI * 2 * typeIndex / typeOrder.length) - Math.PI / 2;
        const gx = centerX + Math.cos(angle) * radius;
        const gy = centerY + Math.sin(angle) * radius;
        group.forEach((node, index) => {
          const local = Math.PI * 2 * index / Math.max(1, group.length);
          const spread = 34 + group.length * 2;
          node.x = gx + Math.cos(local) * spread;
          node.y = gy + Math.sin(local) * spread;
        });
      });
    }

    function shortLabel(value) {
      const text = String(value || '');
      return text.length > 18 ? text.slice(0, 16) + '..' : text;
    }

    window.addEventListener('resize', () => {
      if (state.graph) renderGraph();
    });
    boot().catch(error => {
      document.getElementById('source').textContent = `error: ${error}`;
    });
  </script>
</body>
</html>
"""
