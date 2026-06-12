"""Refresh the redistributable small OA sample from official public APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
STRING_URL = "https://string-db.org/api/json/network"
STRING_VERSION_URL = "https://string-db.org/api/json/version"
ENSEMBL_XREF_URL = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol}"
REACTOME_MAPPING_URL = (
    "https://reactome.org/ContentService/data/mapping/ENSEMBL/{gene_id}/pathways"
)
REACTOME_ANCESTORS_URL = (
    "https://reactome.org/ContentService/data/event/{pathway_id}/ancestors"
)
REACTOME_VERSION_URL = "https://reactome.org/ContentService/data/database/version"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
QUICKGO_ANNOTATION_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
QUICKGO_TERM_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
CHEMBL_STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"
CHEMBL_TARGET_URL = "https://www.ebi.ac.uk/chembl/api/data/target.json"
CHEMBL_MECHANISM_URL = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{drug_id}.json"

OA_EFO = "EFO_0004616"
EXTRA_OA_SYMBOLS = ("MMP13", "ADAMTS5", "COL2A1", "FRZB", "SOX9", "WNT5A")
CHEMBL_TARGET_SYMBOLS = ("NGF", "PTGS2", "MMP13", "ADAMTS5")
PATHWAY_KEYWORDS = (
    "cartilage",
    "collagen",
    "extracellular",
    "matrix",
    "osteo",
    "tgf",
    "bmp",
    "wnt",
    "growth factor",
    "inflamm",
    "eicosa",
)
EVIDENCE_PRIORITY = {
    "IDA": 0,
    "IMP": 1,
    "IGI": 2,
    "IPI": 3,
    "IEP": 4,
    "TAS": 5,
    "IC": 6,
    "ISS": 7,
    "IEA": 8,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--retrieval-date", required=True)
    args = parser.parse_args()
    build_snapshot(args.output_dir, retrieval_date=args.retrieval_date)


def build_snapshot(output_dir: Path, *, retrieval_date: str) -> None:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "kgtp-sample-snapshot/1.0"})
    disease, selected_targets, all_targets = _open_targets(session)
    string_rows, string_version = _string_network(
        session, [target["symbol"] for target in selected_targets]
    )

    gene_records = {
        target["symbol"]: {
            "node_id": target["gene_id"],
            "node_type": "Gene",
            "label": target["name"],
            "symbol": target["symbol"],
            "description": target["name"],
        }
        for target in selected_targets
    }
    for symbol in sorted(_string_symbols(string_rows)):
        if symbol in gene_records:
            continue
        gene_id = _ensembl_gene_id(session, symbol)
        if gene_id is None:
            continue
        gene_records[symbol] = {
            "node_id": gene_id,
            "node_type": "Gene",
            "label": symbol,
            "symbol": symbol,
            "description": f"Human gene {symbol}",
        }

    gene_gene = _string_edges(string_rows, gene_records)
    reactome_version = _get_text(session, REACTOME_VERSION_URL)
    gene_pathway, pathway_hierarchy, pathway_nodes = _reactome_edges(
        session, gene_records
    )
    uniprot_release, gene_go, go_nodes = _go_edges(session, gene_records)
    chembl_status = _get_json(session, CHEMBL_STATUS_URL)
    drug_gene, drug_nodes = _chembl_edges(session, gene_records)

    nodes = pd.DataFrame(
        [
            {
                "node_id": disease["id"],
                "node_type": "Disease",
                "label": disease["name"],
                "symbol": "",
                "description": "Knee osteoarthritis disease anchor.",
                "smiles": "",
                "namespace": "",
            },
            *gene_records.values(),
            *pathway_nodes.values(),
            *drug_nodes.values(),
            *go_nodes.values(),
        ]
    )
    nodes = _with_node_defaults(nodes)
    disease_gene = pd.DataFrame(
        [
            {
                "disease_id": disease["id"],
                "gene_id": target["gene_id"],
                "score": target["score"],
            }
            for target in selected_targets
        ]
    )

    tables = {
        "nodes.parquet": _sorted(nodes, ["node_type", "node_id"]),
        "disease_gene.parquet": _sorted(disease_gene, ["disease_id", "gene_id"]),
        "gene_gene.parquet": _sorted(gene_gene, ["gene_a", "gene_b"]),
        "gene_pathway.parquet": _sorted(gene_pathway, ["gene_id", "pathway_id"]),
        "drug_gene.parquet": _sorted(drug_gene, ["drug_id", "gene_id"]),
        "gene_go.parquet": _sorted(gene_go, ["gene_id", "go_id"]),
        "pathway_pathway.parquet": _sorted(
            pathway_hierarchy, ["parent_pathway_id", "child_pathway_id"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_parquet(output_dir / filename, index=False)

    manifest = {
        "schema_version": 1,
        "dataset": "kgtp-small-oa-sample",
        "retrieval_date": retrieval_date,
        "disease_anchor": disease,
        "selection": {
            "open_targets_total_associated_targets": len(all_targets),
            "positive_edges": (
                "Top 24 Open Targets associations plus six prespecified "
                "OA sanity-check genes when present."
            ),
            "string_expansion": (
                "STRING high-confidence network with up to 30 added neighbors; "
                "only nodes mapped to Ensembl genes were retained."
            ),
        },
        "sources": {
            "open_targets": {
                "release": "live API snapshot; release not exposed by response",
                "url": OPEN_TARGETS_URL,
                "license": "CC0-1.0",
            },
            "string": {
                "release": string_version,
                "url": STRING_URL,
                "license": "CC-BY-4.0",
            },
            "reactome": {
                "release": reactome_version,
                "url": REACTOME_MAPPING_URL,
                "license": "CC0-1.0 for database data",
            },
            "chembl": {
                "release": str(chembl_status["chembl_db_version"]),
                "release_date": str(chembl_status["chembl_release_date"]),
                "url": CHEMBL_MECHANISM_URL,
                "license": "CC-BY-SA-3.0",
            },
            "uniprot": {
                "release": uniprot_release,
                "url": UNIPROT_SEARCH_URL,
                "license": "CC-BY-4.0",
            },
            "goa_quickgo": {
                "release": "live QuickGO snapshot",
                "url": QUICKGO_ANNOTATION_URL,
                "license": "CC-BY-4.0",
            },
            "ensembl": {
                "release": "live REST snapshot",
                "url": ENSEMBL_XREF_URL,
                "license": "CC-BY-4.0",
            },
        },
        "files": {
            filename: {
                "sha256": _sha256(output_dir / filename),
                "bytes": (output_dir / filename).stat().st_size,
                "rows": len(table),
                "columns": list(table.columns),
            }
            for filename, table in tables.items()
        },
        "transformations": [
            "Normalized human gene identifiers to versionless Ensembl gene IDs.",
            "Canonicalized STRING gene pairs lexicographically.",
            "Scaled STRING combined scores from 0-1 to integer 0-1000.",
            "Selected at most two Reactome pathways per gene, preferring OA-relevant labels.",
            "Selected at most two non-NOT GOA annotations per mapped UniProt accession.",
            "Selected at most two ChEMBL mechanisms per single-protein target.",
            "Sorted every output table deterministically and removed duplicate keys.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _open_targets(
    session: requests.Session,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    query = """
    query DiseaseTargets($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: 2000}) {
          rows {
            score
            target { id approvedSymbol approvedName }
          }
        }
      }
    }
    """
    payload = _post_json(
        session,
        OPEN_TARGETS_URL,
        json={"query": query, "variables": {"efoId": OA_EFO}},
    )
    disease_payload = payload["data"]["disease"]
    rows = disease_payload["associatedTargets"]["rows"]
    all_targets = [
        {
            "gene_id": row["target"]["id"],
            "symbol": row["target"]["approvedSymbol"],
            "name": row["target"]["approvedName"],
            "score": float(row["score"]),
        }
        for row in rows
    ]
    by_symbol = {target["symbol"]: target for target in all_targets}
    selected = list(all_targets[:24])
    selected_ids = {target["gene_id"] for target in selected}
    for symbol in EXTRA_OA_SYMBOLS:
        target = by_symbol.get(symbol)
        if target is not None and target["gene_id"] not in selected_ids:
            selected.append(target)
            selected_ids.add(target["gene_id"])
    selected.sort(key=lambda row: (-float(row["score"]), str(row["gene_id"])))
    disease = {"id": disease_payload["id"], "name": disease_payload["name"]}
    return disease, selected, all_targets


def _string_network(
    session: requests.Session, symbols: list[str]
) -> tuple[list[dict[str, Any]], str]:
    version_payload = _get_json(session, STRING_VERSION_URL)
    rows = _post_json(
        session,
        STRING_URL,
        data={
            "identifiers": "\r".join(symbols),
            "species": 9606,
            "required_score": 700,
            "add_nodes": 30,
        },
    )
    return rows, str(version_payload[0]["string_version"])


def _string_symbols(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(row[key]) for row in rows for key in ("preferredName_A", "preferredName_B")
    }


def _ensembl_gene_id(session: requests.Session, symbol: str) -> str | None:
    rows = _get_json(
        session,
        ENSEMBL_XREF_URL.format(symbol=symbol),
        params={"external_db": "HGNC"},
        headers={"Content-Type": "application/json"},
    )
    ids = sorted(
        str(row["id"])
        for row in rows
        if row.get("type") == "gene" and str(row.get("id", "")).startswith("ENSG")
    )
    return ids[0] if ids else None


def _string_edges(
    rows: list[dict[str, Any]], gene_records: dict[str, dict[str, str]]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        symbol_a = str(row["preferredName_A"])
        symbol_b = str(row["preferredName_B"])
        if symbol_a not in gene_records or symbol_b not in gene_records:
            continue
        gene_a, gene_b = sorted(
            (gene_records[symbol_a]["node_id"], gene_records[symbol_b]["node_id"])
        )
        records.append(
            {
                "gene_a": gene_a,
                "gene_b": gene_b,
                "score": round(float(row["score"]) * 1000),
                "string_protein_a": str(row["stringId_A"]),
                "string_protein_b": str(row["stringId_B"]),
            }
        )
    return pd.DataFrame(records).drop_duplicates(["gene_a", "gene_b"])


def _reactome_edges(
    session: requests.Session,
    gene_records: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    gene_pathway_records: list[dict[str, str]] = []
    hierarchy_records: list[dict[str, str]] = []
    pathway_nodes: dict[str, dict[str, str]] = {}
    for symbol in sorted(gene_records):
        gene_id = gene_records[symbol]["node_id"]
        pathways = _get_json_or_default(
            session,
            REACTOME_MAPPING_URL.format(gene_id=gene_id),
            default=[],
        )
        human = [
            pathway
            for pathway in pathways
            if pathway.get("speciesName") == "Homo sapiens"
            and str(pathway.get("stId", "")).startswith("R-HSA-")
        ]
        human.sort(
            key=lambda pathway: (
                not any(
                    keyword in str(pathway["displayName"]).lower()
                    for keyword in PATHWAY_KEYWORDS
                ),
                str(pathway["stId"]),
            )
        )
        for pathway in human[:2]:
            pathway_id = str(pathway["stId"])
            pathway_name = str(pathway["displayName"])
            gene_pathway_records.append(
                {
                    "gene_id": gene_id,
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                }
            )
            pathway_nodes[pathway_id] = _pathway_node(pathway_id, pathway_name)
            ancestor_paths = _get_json_or_default(
                session,
                REACTOME_ANCESTORS_URL.format(pathway_id=pathway_id),
                default=[],
            )
            for ancestor_path in ancestor_paths:
                if len(ancestor_path) < 2:
                    continue
                child = ancestor_path[0]
                parent = ancestor_path[1]
                child_id = str(child["stId"])
                parent_id = str(parent["stId"])
                hierarchy_records.append(
                    {
                        "parent_pathway_id": parent_id,
                        "child_pathway_id": child_id,
                    }
                )
                pathway_nodes[child_id] = _pathway_node(
                    child_id, str(child["displayName"])
                )
                pathway_nodes[parent_id] = _pathway_node(
                    parent_id, str(parent["displayName"])
                )
    gene_pathway = pd.DataFrame(gene_pathway_records).drop_duplicates(
        ["gene_id", "pathway_id"]
    )
    hierarchy = pd.DataFrame(hierarchy_records).drop_duplicates(
        ["parent_pathway_id", "child_pathway_id"]
    )
    return gene_pathway, hierarchy, pathway_nodes


def _pathway_node(pathway_id: str, name: str) -> dict[str, str]:
    return {
        "node_id": pathway_id,
        "node_type": "Pathway",
        "label": name,
        "symbol": "",
        "description": name,
        "smiles": "",
        "namespace": "",
    }


def _go_edges(
    session: requests.Session,
    gene_records: dict[str, dict[str, str]],
) -> tuple[str, pd.DataFrame, dict[str, dict[str, str]]]:
    records: list[dict[str, str]] = []
    go_ids: set[str] = set()
    release = "unknown"
    for symbol in sorted(gene_records):
        response = session.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": (
                    f"(gene_exact:{symbol}) AND (organism_id:9606) AND (reviewed:true)"
                ),
                "format": "tsv",
                "fields": "accession",
            },
            timeout=60,
        )
        response.raise_for_status()
        release = response.headers.get("X-UniProt-Release", release)
        lines = [line for line in response.text.splitlines()[1:] if line.strip()]
        if not lines:
            continue
        accession = lines[0].split("\t")[0]
        gene_records[symbol]["uniprot_id"] = accession
        payload = _get_json(
            session,
            QUICKGO_ANNOTATION_URL,
            params={
                "geneProductId": f"UniProtKB:{accession}",
                "taxonId": 9606,
                "limit": 100,
                "page": 1,
            },
            headers={"Accept": "application/json"},
        )
        annotations = [
            row
            for row in payload.get("results", [])
            if "NOT" not in str(row.get("qualifier", "")).upper()
        ]
        annotations.sort(
            key=lambda row: (
                EVIDENCE_PRIORITY.get(str(row.get("goEvidence", "")), 99),
                str(row["goId"]),
            )
        )
        selected: dict[str, dict[str, Any]] = {}
        for annotation in annotations:
            selected.setdefault(str(annotation["goId"]), annotation)
            if len(selected) == 2:
                break
        for go_id, annotation in selected.items():
            go_ids.add(go_id)
            records.append(
                {
                    "gene_id": gene_records[symbol]["node_id"],
                    "go_id": go_id,
                    "evidence_code": str(annotation.get("goEvidence", "")),
                    "qualifier": str(annotation.get("qualifier", "")),
                    "assigned_by": str(annotation.get("assignedBy", "")),
                    "reference": str(annotation.get("reference", "")),
                }
            )

    go_nodes: dict[str, dict[str, str]] = {}
    for go_id in sorted(go_ids):
        payload = _get_json(
            session,
            QUICKGO_TERM_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        result = payload["results"][0]
        aspect = str(result.get("aspect", "biological_process"))
        namespace = {
            "biological_process": "BP",
            "molecular_function": "MF",
            "cellular_component": "CC",
        }.get(aspect, aspect)
        name = str(result["name"])
        go_nodes[go_id] = {
            "node_id": go_id,
            "node_type": "GOTerm",
            "label": name,
            "symbol": "",
            "description": name,
            "smiles": "",
            "namespace": namespace,
        }
    return release, pd.DataFrame(records), go_nodes


def _chembl_edges(
    session: requests.Session,
    gene_records: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    edge_records: list[dict[str, str]] = []
    drug_nodes: dict[str, dict[str, str]] = {}
    for symbol in CHEMBL_TARGET_SYMBOLS:
        gene = gene_records.get(symbol)
        if gene is None or not gene.get("uniprot_id"):
            continue
        targets = _get_json(
            session,
            CHEMBL_TARGET_URL,
            params={
                "target_components__accession": gene["uniprot_id"],
                "limit": 20,
            },
        ).get("targets", [])
        single_targets = sorted(
            (
                target
                for target in targets
                if target.get("target_type") == "SINGLE PROTEIN"
            ),
            key=lambda target: str(target["target_chembl_id"]),
        )
        if not single_targets:
            continue
        target_id = str(single_targets[0]["target_chembl_id"])
        mechanisms = _get_json(
            session,
            CHEMBL_MECHANISM_URL,
            params={"target_chembl_id": target_id, "limit": 50},
        ).get("mechanisms", [])
        mechanisms.sort(key=lambda row: str(row["molecule_chembl_id"]))
        for mechanism in mechanisms[:2]:
            drug_id = str(mechanism["molecule_chembl_id"])
            edge_records.append(
                {
                    "drug_id": drug_id,
                    "gene_id": gene["node_id"],
                    "target_chembl_id": target_id,
                    "action_type": str(mechanism.get("action_type", "")),
                    "mechanism_of_action": str(
                        mechanism.get("mechanism_of_action", "")
                    ),
                }
            )
            molecule = _get_json(session, CHEMBL_MOLECULE_URL.format(drug_id=drug_id))
            structures = molecule.get("molecule_structures") or {}
            name = str(molecule.get("pref_name") or drug_id)
            drug_nodes[drug_id] = {
                "node_id": drug_id,
                "node_type": "Drug",
                "label": name,
                "symbol": "",
                "description": name,
                "smiles": str(structures.get("canonical_smiles") or ""),
                "namespace": "",
            }
    return pd.DataFrame(edge_records), drug_nodes


def _with_node_defaults(nodes: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "label",
        "symbol",
        "description",
        "smiles",
        "namespace",
        "uniprot_id",
    ):
        if column not in nodes.columns:
            nodes[column] = ""
        nodes[column] = nodes[column].fillna("").astype(str)
    return nodes[
        [
            "node_id",
            "node_type",
            "label",
            "symbol",
            "description",
            "smiles",
            "namespace",
            "uniprot_id",
        ]
    ]


def _sorted(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table.drop_duplicates(columns).sort_values(columns).reset_index(drop=True)


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    response = session.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _post_json(
    session: requests.Session,
    url: str,
    *,
    data: dict[str, object] | None = None,
    json: dict[str, object] | None = None,
) -> Any:
    response = session.post(url, data=data, json=json, timeout=120)
    response.raise_for_status()
    return response.json()


def _get_json_or_default(
    session: requests.Session,
    url: str,
    *,
    default: Any,
) -> Any:
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return default
    response.raise_for_status()
    return response.json()


def _get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.text.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
