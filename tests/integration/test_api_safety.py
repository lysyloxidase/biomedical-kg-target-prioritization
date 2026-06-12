from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kgtp.api.app import app, get_service
from kgtp.artifacts import ArtifactPaths


def _configure(monkeypatch: pytest.MonkeyPatch, paths: ArtifactPaths) -> None:
    values = {
        "KGTP_CHECKPOINT_PATH": paths.checkpoint,
        "KGTP_MODEL_CONFIG_PATH": paths.model_config,
        "KGTP_GRAPH_PATH": paths.graph,
        "KGTP_DATASET_MANIFEST_PATH": paths.dataset_manifest,
        "KGTP_GRAPH_MANIFEST_PATH": paths.graph_manifest,
        "KGTP_FEATURE_MANIFEST_PATH": paths.feature_manifest,
        "KGTP_NODE_INDEX_MAP_PATH": paths.node_index_map,
        "KGTP_SPLIT_METADATA_PATH": paths.split_metadata,
        "KGTP_VALIDATION_METRICS_PATH": paths.validation_metrics,
        "KGTP_RUN_MANIFEST_PATH": paths.run_manifest,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("KGTP_DEMO_MODE", raising=False)
    get_service.cache_clear()


def test_valid_trained_model_serves_predictions(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
) -> None:
    _configure(monkeypatch, trained_artifact_paths)
    client = TestClient(app)

    health = client.get("/health")
    prediction = client.get(
        "/predict",
        params={"disease": "EFO_0004616", "top_k": 2},
    )

    assert health.status_code == 200, health.json()
    assert health.json()["trained"] is True
    assert prediction.status_code == 200
    payload = prediction.json()
    assert payload["trained"] is True
    assert payload["run_id"] == "test-run-13"
    assert payload["checkpoint_sha256"]
    assert payload["dataset_manifest_sha256"]
    assert payload["candidate_protocol"] == "full_candidate_all_eligible_genes"
    assert payload["hypothesis_only"] is True
    assert len(payload["results"]) == 2
    assert payload["results"][0]["explanation_subgraph"]["model_attribution"] is True


def test_missing_checkpoint_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    paths = ArtifactPaths(
        **{
            **vars(trained_artifact_paths),
            "checkpoint": tmp_path / "missing.pt",
        }
    )
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "checkpoint is missing" in response.json()["detail"]


def test_corrupted_checkpoint_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    corrupted = tmp_path / "corrupted.pt"
    corrupted.write_bytes(b"not a torch checkpoint")
    paths = ArtifactPaths(**{**vars(trained_artifact_paths), "checkpoint": corrupted})
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "Checkpoint loading failed" in response.json()["detail"]


def test_untrained_model_metadata_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    config = json.loads(trained_artifact_paths.model_config.read_text(encoding="utf-8"))
    config["trained"] = False
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    paths = ArtifactPaths(**{**vars(trained_artifact_paths), "model_config": changed})
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "not marked as trained" in response.json()["detail"]


def test_missing_validation_metadata_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    metrics = json.loads(
        trained_artifact_paths.validation_metrics.read_text(encoding="utf-8")
    )
    metrics.pop("metrics")
    changed = tmp_path / "metrics.json"
    changed.write_text(json.dumps(metrics), encoding="utf-8")
    paths = ArtifactPaths(
        **{**vars(trained_artifact_paths), "validation_metrics": changed}
    )
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "contains no metrics" in response.json()["detail"]


def test_graph_hash_mismatch_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        trained_artifact_paths.dataset_manifest.read_text(encoding="utf-8")
    )
    manifest["full_reference_graph_hash"] = "wrong-graph"
    changed = tmp_path / "dataset.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    paths = ArtifactPaths(
        **{**vars(trained_artifact_paths), "dataset_manifest": changed}
    )
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "dataset full-reference graph hash" in response.json()["detail"]


def test_feature_mismatch_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        trained_artifact_paths.feature_manifest.read_text(encoding="utf-8")
    )
    manifest["metadata"]["fitted_graph_hash"] = "wrong-train-graph"
    changed = tmp_path / "features.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    paths = ArtifactPaths(
        **{**vars(trained_artifact_paths), "feature_manifest": changed}
    )
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "feature fitting graph" in response.json()["detail"]


def test_run_manifest_mismatch_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
    tmp_path: Path,
) -> None:
    run = json.loads(trained_artifact_paths.run_manifest.read_text(encoding="utf-8"))
    run["checkpoint_hash"] = "wrong-checkpoint"
    changed = tmp_path / "run.json"
    changed.write_text(json.dumps(run), encoding="utf-8")
    paths = ArtifactPaths(**{**vars(trained_artifact_paths), "run_manifest": changed})
    _configure(monkeypatch, paths)

    response = TestClient(app).get("/predict")

    assert response.status_code == 503
    assert "Run manifest mismatch for checkpoint_hash" in response.json()["detail"]


def test_demo_mode_is_explicitly_marked_non_scientific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KGTP_DEMO_MODE", "true")
    get_service.cache_clear()

    response = TestClient(app).get(
        "/predict",
        params={"disease": "EFO_0004616", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "synthetic-smoke-demo"
    assert any("DEMO MODE" in warning for warning in payload["warnings"])


def test_unknown_disease_and_invalid_top_k(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_paths: ArtifactPaths,
) -> None:
    _configure(monkeypatch, trained_artifact_paths)
    client = TestClient(app)

    unknown = client.get("/predict", params={"disease": "MISSING", "top_k": 1})
    invalid = client.get("/predict", params={"top_k": 0})

    assert unknown.status_code == 404
    assert invalid.status_code == 422
