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
        "Headline Results",
        "Ablations",
        "Graph Statistics",
        "Interpretability Case Study",
        "Honest Findings",
        "Data Sources And Licenses",
        "Reproducibility",
    ):
        assert section in readme
    assert "AUPRC" in readme
    assert "Computational hypothesis" in readme
    assert "pending full OA run" in readme
    assert "tests/unit/test_splits.py" in ci
    assert "smoke-train" in ci
    assert "docker build" in ci
    assert "pytest --cov=kgtp" in ci
    assert "mkdocs gh-deploy" in docs
    assert "uvicorn" in dockerfile
    assert "## [1.0.0] - 2026-06-11" in changelog


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
