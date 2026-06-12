"""Export normalized KG tables or Neo4j rows to PyG ``HeteroData``."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike, read_table
from kgtp.kg.neo4j_loader import Neo4jConfig

NODE_TYPE_TO_PYG = {
    "Disease": "disease",
    "Gene": "gene",
    "Pathway": "pathway",
    "Drug": "drug",
    "GOTerm": "go_term",
}

PYG_TO_NODE_TYPE = {value: key for key, value in NODE_TYPE_TO_PYG.items()}

REVERSE_EDGE_TYPES: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("disease", "associated_with", "gene"): (
        "gene",
        "rev_associated_with",
        "disease",
    ),
    ("drug", "targets", "gene"): ("gene", "rev_targets", "drug"),
    ("gene", "participates_in", "pathway"): (
        "pathway",
        "rev_participates_in",
        "gene",
    ),
    ("gene", "annotated_with", "go_term"): (
        "go_term",
        "rev_annotated_with",
        "gene",
    ),
    ("pathway", "parent_of", "pathway"): (
        "pathway",
        "rev_parent_of",
        "pathway",
    ),
}


def build_heterodata(
    nodes: pd.DataFrame | None = None,
    edge_tables: Mapping[str, pd.DataFrame] | None = None,
    *,
    processed_dir: PathLike = "data/processed",
    output_dir: PathLike | None = None,
    gene_feature_mode: str = "structural",
    add_reverse_edges: bool = True,
    precomputed_features: Mapping[str, torch.Tensor] | None = None,
) -> HeteroData:
    """Build a PyG ``HeteroData`` object from deterministic node/edge tables."""

    if nodes is None or edge_tables is None:
        nodes, edge_tables = read_processed_graph(processed_dir)

    normalized_nodes = _normalize_node_table(nodes)
    data = HeteroData()
    node_maps = _build_node_maps(normalized_nodes)

    for node_type in NODE_TYPE_TO_PYG.values():
        ids = list(node_maps[node_type])
        data[node_type].num_nodes = len(ids)
        data[node_type].node_id = ids

    generated_features: dict[str, Callable[[], torch.Tensor]] = {
        "gene": lambda: _gene_features(
            list(node_maps["gene"]),
            normalized_nodes,
            edge_tables,
            mode=gene_feature_mode,
        ),
        "disease": lambda: _disease_features(
            list(node_maps["disease"]), normalized_nodes
        ),
        "drug": lambda: _drug_features(list(node_maps["drug"]), normalized_nodes),
        "pathway": lambda: _pathway_features(
            list(node_maps["pathway"]),
            normalized_nodes,
            edge_tables,
        ),
        "go_term": lambda: _go_term_features(
            list(node_maps["go_term"]),
            normalized_nodes,
            edge_tables,
        ),
    }
    for node_type, build_features in generated_features.items():
        features = (
            precomputed_features[node_type]
            if precomputed_features is not None and node_type in precomputed_features
            else build_features()
        )
        if features.size(0) != len(node_maps[node_type]):
            msg = (
                f"Feature row count for {node_type} is {features.size(0)}, "
                f"expected {len(node_maps[node_type])}"
            )
            raise ValueError(msg)
        data[node_type].x = features

    for edges in edge_tables.values():
        edge_type = _edge_type_from_table(edges)
        edge_index = _edge_index(edges, edge_type, node_maps)
        data[edge_type].edge_index = edge_index
        if add_reverse_edges and edge_type in REVERSE_EDGE_TYPES:
            data[REVERSE_EDGE_TYPES[edge_type]].edge_index = edge_index.flip(0)

    data.node_index_maps = node_maps
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_node_index_maps(node_maps, output / "node_index_maps.json")
        torch.save(data, output / "heterodata.pt")
    return data


def read_processed_graph(
    processed_dir: PathLike = "data/processed",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read Phase 1 processed parquet tables."""

    processed = Path(processed_dir)
    nodes = read_table(processed / "nodes.parquet")
    edge_tables = {
        path.stem: read_table(path)
        for path in sorted((processed / "edges").glob("*.parquet"))
    }
    return nodes, edge_tables


def fetch_neo4j_graph(
    config: Neo4jConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fetch Neo4j nodes/relationships with deterministic ordering."""

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    try:
        with driver.session(database=config.database) as session:
            node_rows = session.run(
                "MATCH (n) "
                "RETURN labels(n)[0] AS node_type, n.id AS node_id, properties(n) AS properties "
                "ORDER BY node_type, node_id"
            )
            nodes = _rows_with_properties(node_rows, ["node_type", "node_id"])

            edge_rows = session.run(
                "MATCH (s)-[r]->(t) "
                "RETURN labels(s)[0] AS source_type, s.id AS source_id, "
                "type(r) AS edge_type, labels(t)[0] AS target_type, "
                "t.id AS target_id, properties(r) AS properties "
                "ORDER BY source_type, edge_type, target_type, source_id, target_id"
            )
            edges = _rows_with_properties(
                edge_rows,
                ["source_type", "source_id", "edge_type", "target_type", "target_id"],
            )
    finally:
        driver.close()

    edge_tables: dict[str, pd.DataFrame] = {}
    if not edges.empty:
        edges["edge_type"] = edges["edge_type"].astype(str).str.lower()
        for edge_type, group in edges.groupby("edge_type", sort=True):
            edge_tables[str(edge_type)] = group.reset_index(drop=True)
    return nodes, edge_tables


def build_heterodata_from_neo4j(
    config: Neo4jConfig,
    *,
    output_dir: PathLike | None = None,
    gene_feature_mode: str = "structural",
) -> HeteroData:
    """Fetch a loaded Neo4j KG and convert it to ``HeteroData``."""

    nodes, edge_tables = fetch_neo4j_graph(config)
    return build_heterodata(
        nodes,
        edge_tables,
        output_dir=output_dir,
        gene_feature_mode=gene_feature_mode,
    )


def write_node_index_maps(
    node_maps: Mapping[str, Mapping[str, int]],
    path: PathLike,
) -> None:
    """Persist node-to-row maps for reproducible interpretation lookups."""

    serializable = {
        node_type: dict(sorted(mapping.items(), key=lambda item: item[1]))
        for node_type, mapping in sorted(node_maps.items())
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )


def _normalize_node_table(nodes: pd.DataFrame) -> pd.DataFrame:
    normalized = nodes.copy()
    normalized["node_type"] = normalized["node_type"].map(
        lambda value: NODE_TYPE_TO_PYG.get(str(value), str(value).lower())
    )
    normalized["node_id"] = normalized["node_id"].astype(str)
    return normalized.sort_values(["node_type", "node_id"]).reset_index(drop=True)


def _build_node_maps(nodes: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for node_type in NODE_TYPE_TO_PYG.values():
        ids = sorted(
            nodes.loc[nodes["node_type"] == node_type, "node_id"].astype(str).unique()
        )
        maps[node_type] = {node_id: index for index, node_id in enumerate(ids)}
    return maps


def _edge_type_from_table(edges: pd.DataFrame) -> tuple[str, str, str]:
    if edges.empty:
        msg = "Cannot infer edge type from an empty edge table"
        raise ValueError(msg)
    row = edges.iloc[0]
    source_type = NODE_TYPE_TO_PYG.get(
        str(row["source_type"]), str(row["source_type"]).lower()
    )
    relation = str(row["edge_type"]).lower()
    target_type = NODE_TYPE_TO_PYG.get(
        str(row["target_type"]), str(row["target_type"]).lower()
    )
    return source_type, relation, target_type


def _edge_index(
    edges: pd.DataFrame,
    edge_type: tuple[str, str, str],
    node_maps: Mapping[str, Mapping[str, int]],
) -> torch.Tensor:
    source_type, _, target_type = edge_type
    source_map = node_maps[source_type]
    target_map = node_maps[target_type]
    pairs: list[tuple[int, int]] = []
    for source_id, target_id in zip(
        edges["source_id"], edges["target_id"], strict=False
    ):
        source_key = str(source_id)
        target_key = str(target_id)
        if source_key in source_map and target_key in target_map:
            pairs.append((source_map[source_key], target_map[target_key]))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _gene_features(
    gene_ids: list[str],
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    mode: str,
) -> torch.Tensor:
    if not gene_ids:
        return torch.empty((0, 2), dtype=torch.float32)

    if mode == "none":
        return torch.eye(len(gene_ids), dtype=torch.float32)

    structural = _gene_structural_features(gene_ids, edge_tables)
    if mode == "go":
        return torch.cat([structural, _go_multihot(gene_ids, edge_tables)], dim=1)

    embeddings = _optional_vector_column(
        nodes, "gene", gene_ids, ("esm2_embedding", "embedding")
    )
    if embeddings is not None:
        return torch.cat([structural, embeddings], dim=1)
    return structural


def _gene_structural_features(
    gene_ids: list[str],
    edge_tables: Mapping[str, pd.DataFrame],
) -> torch.Tensor:
    degree: defaultdict[str, int] = defaultdict(int)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for edges in edge_tables.values():
        for source_id, target_id in zip(
            edges["source_id"], edges["target_id"], strict=False
        ):
            source = str(source_id)
            target = str(target_id)
            degree[source] += 1
            degree[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)

    pagerank = _pagerank(adjacency)
    rows = [[float(degree[gene]), float(pagerank.get(gene, 0.0))] for gene in gene_ids]
    return torch.tensor(rows, dtype=torch.float32)


def _pagerank(
    adjacency: Mapping[str, set[str]],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    nodes = sorted(adjacency)
    if not nodes:
        return {}
    initial = 1.0 / len(nodes)
    scores = dict.fromkeys(nodes, initial)
    for _ in range(iterations):
        updated = {node: (1.0 - damping) / len(nodes) for node in nodes}
        for source in nodes:
            neighbors = adjacency[source]
            if not neighbors:
                continue
            contribution = damping * scores[source] / len(neighbors)
            for target in neighbors:
                updated[target] += contribution
        scores = updated
    return scores


def _go_multihot(
    gene_ids: list[str],
    edge_tables: Mapping[str, pd.DataFrame],
    *,
    max_terms: int = 512,
) -> torch.Tensor:
    gene_go = _find_edge_table(edge_tables, ("Gene", "annotated_with", "GOTerm"))
    if gene_go is None or gene_go.empty:
        return torch.empty((len(gene_ids), 0), dtype=torch.float32)
    terms = sorted(gene_go["target_id"].astype(str).unique())[:max_terms]
    term_index = {term: index for index, term in enumerate(terms)}
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    out = torch.zeros((len(gene_ids), len(terms)), dtype=torch.float32)
    for gene_id, go_id in zip(gene_go["source_id"], gene_go["target_id"], strict=False):
        gene = str(gene_id)
        term = str(go_id)
        if gene in gene_index and term in term_index:
            out[gene_index[gene], term_index[term]] = 1.0
    return out


def _disease_features(disease_ids: list[str], nodes: pd.DataFrame) -> torch.Tensor:
    embeddings = _optional_vector_column(
        nodes, "disease", disease_ids, ("text_embedding", "embedding")
    )
    if embeddings is not None:
        return embeddings
    return _identity_or_empty(len(disease_ids))


def _drug_features(drug_ids: list[str], nodes: pd.DataFrame) -> torch.Tensor:
    if not drug_ids:
        return torch.empty((0, 2048), dtype=torch.float32)
    drug_rows = _rows_by_id(nodes, "drug")
    rows = []
    for drug_id in drug_ids:
        row = drug_rows.get(drug_id, {})
        smiles = str(
            row.get("smiles")
            or row.get("canonical_smiles")
            or row.get("isomeric_smiles")
            or drug_id
        )
        rows.append(_morgan_fingerprint(smiles))
    return torch.stack(rows).to(torch.float32)


def _morgan_fingerprint(
    smiles: str, *, radius: int = 2, n_bits: int = 2048
) -> torch.Tensor:
    try:
        chem = importlib.import_module("rdkit.Chem")
        all_chem = importlib.import_module("rdkit.Chem.AllChem")
    except ModuleNotFoundError:
        return _hashed_fingerprint(smiles, n_bits=n_bits)

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return _hashed_fingerprint(smiles, n_bits=n_bits)
    fingerprint = all_chem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )
    bits = [float(bit) for bit in fingerprint.ToBitString()]
    return torch.tensor(bits, dtype=torch.float32)


def _hashed_fingerprint(
    value: str, *, n_bits: int = 2048, n_hashes: int = 32
) -> torch.Tensor:
    out = torch.zeros(n_bits, dtype=torch.float32)
    text = value or "missing"
    for salt in range(n_hashes):
        digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % n_bits
        out[index] = 1.0
    return out


def _pathway_features(
    pathway_ids: list[str],
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
) -> torch.Tensor:
    if not pathway_ids:
        return torch.empty((0, 2), dtype=torch.float32)
    participant_counts: defaultdict[str, int] = defaultdict(int)
    gene_pathway = _find_edge_table(edge_tables, ("Gene", "participates_in", "Pathway"))
    if gene_pathway is not None:
        for pathway_id in gene_pathway["target_id"].astype(str):
            participant_counts[pathway_id] += 1

    depths = _pathway_depths(edge_tables)
    rows = [
        [float(participant_counts[pathway]), float(depths.get(pathway, 0))]
        for pathway in pathway_ids
    ]
    return torch.tensor(rows, dtype=torch.float32)


def _pathway_depths(edge_tables: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    hierarchy = _find_edge_table(edge_tables, ("Pathway", "parent_of", "Pathway"))
    if hierarchy is None or hierarchy.empty:
        return {}
    children: defaultdict[str, set[str]] = defaultdict(set)
    parents: defaultdict[str, set[str]] = defaultdict(set)
    pathways: set[str] = set()
    for parent_id, child_id in zip(
        hierarchy["source_id"], hierarchy["target_id"], strict=False
    ):
        parent = str(parent_id)
        child = str(child_id)
        children[parent].add(child)
        parents[child].add(parent)
        pathways.update((parent, child))

    roots = sorted(pathway for pathway in pathways if not parents[pathway])
    depths: dict[str, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        node, depth = queue.popleft()
        if node in depths and depths[node] <= depth:
            continue
        depths[node] = depth
        queue.extend((child, depth + 1) for child in sorted(children[node]))
    return depths


def _go_term_features(
    go_ids: list[str],
    nodes: pd.DataFrame,
    edge_tables: Mapping[str, pd.DataFrame],
) -> torch.Tensor:
    if not go_ids:
        return torch.empty((0, 4), dtype=torch.float32)
    rows_by_id = _rows_by_id(nodes, "go_term")
    annotation_counts: defaultdict[str, int] = defaultdict(int)
    gene_go = _find_edge_table(edge_tables, ("Gene", "annotated_with", "GOTerm"))
    total_annotations = 0
    if gene_go is not None:
        for go_id in gene_go["target_id"].astype(str):
            annotation_counts[go_id] += 1
            total_annotations += 1

    rows: list[list[float]] = []
    for go_id in go_ids:
        row = rows_by_id.get(go_id, {})
        namespace = _normalize_go_namespace(row.get("namespace") or row.get("aspect"))
        one_hot = [1.0 if namespace == value else 0.0 for value in ("BP", "MF", "CC")]
        information_content = _information_content(
            row, go_id, annotation_counts, total_annotations
        )
        rows.append([*one_hot, information_content])
    return torch.tensor(rows, dtype=torch.float32)


def _information_content(
    row: Mapping[str, Any],
    go_id: str,
    annotation_counts: Mapping[str, int],
    total_annotations: int,
) -> float:
    value = row.get("information_content")
    if value is not None and not pd.isna(value):
        return float(value)
    count = annotation_counts.get(go_id, 0)
    if count == 0 or total_annotations == 0:
        return 0.0
    return float(-math.log(count / total_annotations))


def _normalize_go_namespace(value: object) -> str:
    text = str(value or "BP").lower()
    if text in {"p", "bp", "biological_process"}:
        return "BP"
    if text in {"f", "mf", "molecular_function"}:
        return "MF"
    if text in {"c", "cc", "cellular_component"}:
        return "CC"
    return "BP"


def _optional_vector_column(
    nodes: pd.DataFrame,
    node_type: str,
    node_ids: list[str],
    columns: tuple[str, ...],
) -> torch.Tensor | None:
    present = next((column for column in columns if column in nodes.columns), None)
    if present is None:
        return None
    rows = _rows_by_id(nodes, node_type)
    vectors: list[list[float]] = []
    width: int | None = None
    for node_id in node_ids:
        value = rows.get(node_id, {}).get(present)
        vector = _parse_vector(value)
        if vector is None:
            return None
        width = len(vector) if width is None else width
        if len(vector) != width:
            return None
        vectors.append(vector)
    if not vectors:
        return None
    return torch.tensor(vectors, dtype=torch.float32)


def _parse_vector(value: object) -> list[float] | None:
    if value is None or bool(pd.isna(cast(Any, value))):
        return None
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    delimiter = "," if "," in text else " "
    return [float(part) for part in text.split(delimiter) if part.strip()]


def _identity_or_empty(size: int) -> torch.Tensor:
    if size == 0:
        return torch.empty((0, 1), dtype=torch.float32)
    return torch.eye(size, dtype=torch.float32)


def _find_edge_table(
    edge_tables: Mapping[str, pd.DataFrame],
    schema: tuple[str, str, str],
) -> pd.DataFrame | None:
    for edges in edge_tables.values():
        if edges.empty:
            continue
        row = edges.iloc[0]
        triple = (
            str(row["source_type"]),
            str(row["edge_type"]),
            str(row["target_type"]),
        )
        if triple == schema:
            return edges
    return None


def _rows_by_id(nodes: pd.DataFrame, node_type: str) -> dict[str, dict[str, Any]]:
    subset = nodes[nodes["node_type"] == node_type]
    return {
        str(row["node_id"]): cast(dict[str, Any], row.dropna().to_dict())
        for _, row in subset.iterrows()
    }


def _rows_with_properties(rows: Any, base_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {column: row[column] for column in base_columns}
        properties = row.get("properties") or {}
        record.update(properties)
        records.append(record)
    return pd.DataFrame(records)
