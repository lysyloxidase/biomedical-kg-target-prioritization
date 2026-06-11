"""Project configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectPaths(BaseModel):
    """Filesystem paths used by the Phase 1 pipeline."""

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    crosswalk: Path = Path("data/crosswalk")


class Neo4jSettings(BaseModel):
    """Neo4j connection settings."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class GraphSettings(BaseModel):
    """Graph construction settings."""

    disease_efo: str = "EFO_0004616"
    min_nodes: int = 2_000
    max_nodes: int = 6_000
    string_confidence_threshold: int = 700
    max_go_terms: int = 1_000


class Settings(BaseSettings):
    """Runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(env_prefix="KGTP_", env_nested_delimiter="__")

    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping in {path}"
        raise TypeError(msg)
    return data


def load_settings(path: str | Path = "configs/config.yaml") -> Settings:
    """Load settings from `configs/config.yaml` plus environment overrides."""

    data = load_yaml(path)
    return Settings.model_validate(data)
