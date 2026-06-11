from __future__ import annotations

from typing import cast

from fastapi.testclient import TestClient

from kgtp.api.app import app, get_service, graph_stats_from_heterodata, rank_targets
from kgtp.smoke import run_smoke_train, tiny_heterodata


def test_fastapi_health_graph_stats_and_predict_endpoints() -> None:
    get_service.cache_clear()
    client = TestClient(app)

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Heterogeneous biomedical knowledge graph" in dashboard.text
    assert "Target Ranking" in dashboard.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "Computational hypothesis" in health.json()["disclaimer"]

    graph_data = client.get("/graph-data")
    assert graph_data.status_code == 200
    assert graph_data.json()["nodes"]
    assert graph_data.json()["edges"]

    graph_stats = client.get("/graph-stats")
    assert graph_stats.status_code == 200
    payload = graph_stats.json()
    assert payload["node_counts"]["gene"] == 8
    assert payload["positive_disease_gene_links"] == 9

    prediction = client.get("/predict", params={"disease": "EFO_0004616", "top_k": 2})
    assert prediction.status_code == 200
    rows = prediction.json()["results"]
    assert len(rows) == 2
    assert {"gene_id", "gene_symbol", "score", "explanation_subgraph"}.issubset(rows[0])
    assert rows[0]["explanation_subgraph"]["nodes"]

    missing = client.get("/predict", params={"disease": "MISSING", "top_k": 1})
    assert missing.status_code == 404


def test_api_service_helpers_rank_tiny_graph() -> None:
    get_service.cache_clear()
    service = get_service()
    rows = rank_targets(service, 0, top_k=3)
    stats = graph_stats_from_heterodata(tiny_heterodata())

    assert len(rows) == 3
    assert all(row["hypothesis"] is not None for row in rows)
    assert cast(float, stats["mean_degree"]) > 0
    assert cast(float, stats["density"]) > 0


def test_smoke_train_gate_runs_hgt_on_tiny_graph(tmp_path) -> None:
    result = run_smoke_train(max_epochs=1, output_dir=tmp_path)

    assert "AUPRC" in result.metrics
    assert "filtered_MRR" in result.metrics
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()
