"""Idempotent Neo4j loader for normalized KG tables."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd
from neo4j import Query

SAFE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class Neo4jLoader:
    """Load nodes and edges into Neo4j with MERGE semantics."""

    def __init__(self, config: Neo4jConfig) -> None:
        from neo4j import GraphDatabase

        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri, auth=(config.user, config.password)
        )

    def close(self) -> None:
        """Close the underlying Neo4j driver."""

        self.driver.close()

    def create_constraints(self, labels: Iterable[str]) -> None:
        """Create business-key constraints for each node label."""

        with self.driver.session(database=self.config.database) as session:
            for label in sorted(set(labels)):
                safe_label = _safe_name(label)
                query = (
                    f"CREATE CONSTRAINT {safe_label.lower()}_id IF NOT EXISTS "
                    f"FOR (n:{safe_label}) REQUIRE n.id IS UNIQUE"
                )
                session.run(
                    Query(query)  # pyright: ignore[reportArgumentType]
                )

    def load_nodes(self, nodes: pd.DataFrame, *, batch_size: int = 1_000) -> None:
        """Load canonical nodes grouped by label."""

        for node_type, group in nodes.groupby("node_type"):
            label = _safe_name(str(node_type))
            rows = [_node_row(row) for _, row in group.iterrows()]
            query = (
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) "
                "SET n += row.properties"
            )
            self._run_batches(query, rows, batch_size=batch_size)

    def load_edges(
        self,
        edge_tables: Mapping[str, pd.DataFrame],
        *,
        batch_size: int = 1_000,
    ) -> None:
        """Load canonical edges grouped by endpoint labels and relation."""

        for edges in edge_tables.values():
            group_cols = ["source_type", "edge_type", "target_type"]
            for (source_type, edge_type, target_type), group in edges.groupby(
                group_cols
            ):
                source_label = _safe_name(str(source_type))
                target_label = _safe_name(str(target_type))
                relation = _safe_name(str(edge_type).upper())
                rows = [_edge_row(row) for _, row in group.iterrows()]
                query = (
                    "UNWIND $rows AS row "
                    f"MATCH (s:{source_label} {{id: row.source_id}}) "
                    f"MATCH (t:{target_label} {{id: row.target_id}}) "
                    f"MERGE (s)-[r:{relation}]->(t) "
                    "SET r += row.properties"
                )
                self._run_batches(query, rows, batch_size=batch_size)

    def _run_batches(
        self,
        query: str,
        rows: list[dict[str, object]],
        *,
        batch_size: int,
    ) -> None:
        with self.driver.session(database=self.config.database) as session:
            for start in range(0, len(rows), batch_size):
                session.run(
                    Query(query),  # pyright: ignore[reportArgumentType]
                    rows=rows[start : start + batch_size],
                )


def load_graph(
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
    config: Neo4jConfig,
) -> None:
    """Open a loader, create constraints, and load the graph."""

    loader = Neo4jLoader(config)
    try:
        loader.create_constraints(nodes["node_type"].dropna().astype(str))
        loader.load_nodes(nodes)
        loader.load_edges(edge_tables)
    finally:
        loader.close()


def _safe_name(value: str) -> str:
    if SAFE_NAME_RE.fullmatch(value) is None:
        msg = f"Unsafe Neo4j label or relationship type: {value}"
        raise ValueError(msg)
    return value


def _clean_properties(row: pd.Series, skip: set[str]) -> dict[str, object]:
    props: dict[str, object] = {}
    for key, value in row.items():
        if key in skip or pd.isna(value):
            continue
        props[str(key)] = value
    return props


def _node_row(row: pd.Series) -> dict[str, object]:
    return {
        "id": str(row["node_id"]),
        "properties": _clean_properties(row, {"node_id", "node_type"}),
    }


def _edge_row(row: pd.Series) -> dict[str, object]:
    return {
        "source_id": str(row["source_id"]),
        "target_id": str(row["target_id"]),
        "properties": _clean_properties(
            row, {"source_id", "target_id", "source_type", "edge_type", "target_type"}
        ),
    }
