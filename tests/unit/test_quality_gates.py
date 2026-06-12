from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kgtp.data.opentargets import fetch_oa_target_count

ROOT = Path(__file__).resolve().parents[2]


def test_sources_config_pins_six_sources_and_excludes_disgenet() -> None:
    with (ROOT / "configs" / "sources.yaml").open(encoding="utf-8") as handle:
        sources = yaml.safe_load(handle)["sources"]

    included = {
        name for name, spec in sources.items() if spec.get("status") != "EXCLUDED"
    }

    assert included == {
        "open_targets",
        "string",
        "reactome",
        "chembl",
        "uniprot",
        "gene_ontology",
    }
    assert sources["disgenet"]["status"] == "EXCLUDED"
    assert "CC0 substitute" in sources["disgenet"]["reason"]


def test_docs_include_required_adrs_caveats_and_disclaimer() -> None:
    caveats = (ROOT / "docs" / "caveats.md").read_text(encoding="utf-8").splitlines()
    numbered = [line for line in caveats if line[:1].isdigit() and ". " in line]
    adrs = sorted((ROOT / "docs" / "adr").glob("*.md"))

    assert len(numbered) == 14
    assert len(adrs) == 4
    assert (ROOT / "DISCLAIMER.md").exists()


def test_phase7_production_docs_and_ci_gates_are_present() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    for section in (
        "Current Status",
        "Task Definition",
        "Sample Dataset",
        "Pipeline Architecture",
        "Leakage Protection",
        "Implemented Models",
        "Evaluation",
        "Artifacts",
        "Full-Data Prerequisites",
        "Licenses",
        "Limitations",
    ):
        assert section in readme
    assert "AUPRC" in readme
    assert "Computational hypothesis" in readme
    assert "No real full-scale benchmark results" in readme
    assert "test_leakage_prevention.py" in ci
    assert "test_unlabeled_sampling.py" in ci
    assert "test_sample_pipeline.py" in ci
    assert "test_api_safety.py" in ci
    assert "smoke-train" in ci
    assert "docker build" in ci
    assert "pytest --cov=kgtp" in ci
    assert "pip-audit" in ci
    assert "bandit" in ci
    assert "check_coverage.py" in ci
    assert "mkdocs gh-deploy" in docs
    assert "USER kgtp" in dockerfile
    assert "chown -R" not in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["/app/.venv/bin/uvicorn"' in dockerfile
    assert "## [0.2.0] - 2026-06-12" in changelog


def test_required_phase7_docs_and_secret_examples_are_safe() -> None:
    required_docs = (
        "dataset-card.md",
        "model-card.md",
        "evaluation-protocol.md",
        "final-validation-report.md",
        "leakage-prevention.md",
        "reproducibility.md",
        "biomedical-limitations.md",
        "api-safety.md",
    )
    assert all((ROOT / "docs" / name).is_file() for name in required_docs)
    assert (ROOT / "SECURITY.md").is_file()
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    compose = (ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    config = (ROOT / "src" / "kgtp" / "config.py").read_text(encoding="utf-8")
    assert "neo4j/password" not in compose
    assert 'password: str = "password"' not in config
    assert "replace-with-a-strong-local-password" in env_example
    assert ".env" in dockerignore
    assert "artifacts" in dockerignore


def test_github_actions_are_pinned_to_commits() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    uses_lines = [
        line.strip()
        for workflow in workflows
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- uses:")
    ]

    assert uses_lines
    for line in uses_lines:
        reference = line.split("@", maxsplit=1)[1].split(maxsplit=1)[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_open_targets_count_is_read_from_graphql(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "disease": {
                        "id": "EFO_0004616",
                        "associatedTargets": {"count": 321},
                    }
                }
            }

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> Response:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("kgtp.data.opentargets.requests.post", fake_post)

    count = fetch_oa_target_count("https://example.org/graphql", "EFO_0004616")

    assert count == 321
    assert captured["json"]["variables"]["efoId"] == "EFO_0004616"
    assert "associatedTargets" in captured["json"]["query"]
