from __future__ import annotations

from typing import cast

from kgtp.api.app import graph_data_from_heterodata, graph_stats_from_heterodata
from kgtp.smoke import run_smoke_train, tiny_heterodata


def test_api_graph_helpers_describe_tiny_graph() -> None:
    data = tiny_heterodata()
    stats = graph_stats_from_heterodata(data)
    graph = graph_data_from_heterodata(data)

    assert cast(float, stats["mean_degree"]) > 0
    assert cast(float, stats["density"]) > 0
    assert graph["nodes"]
    assert graph["edges"]


def test_smoke_train_gate_runs_hgt_on_tiny_graph(tmp_path) -> None:
    result = run_smoke_train(max_epochs=1, output_dir=tmp_path)

    assert "AUPRC" in result.metrics
    assert "filtered_MRR" in result.metrics
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()
