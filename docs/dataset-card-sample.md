# Dataset Card: Small Knee-OA Sample

## Summary

`data/sample/` is a deliberately small, redistributable snapshot for exercising
the repository end to end. It anchors one disease, knee osteoarthritis
(`EFO_0004616`), and includes public disease-gene, protein interaction, pathway,
drug-target, and Gene Ontology records.

This is not a complete biomedical knowledge graph, a clinical dataset, or a
validated target list. Its purpose is deterministic software testing and
demonstration.

Snapshot retrieval date: **2026-06-12**.

## Files

| File | Rows | Content |
| --- | ---: | --- |
| `nodes.parquet` | 232 | Typed identifiers and selected public attributes |
| `disease_gene.parquet` | 30 | Knee-OA to Ensembl gene associations |
| `gene_gene.parquet` | 195 | High-confidence STRING interactions |
| `gene_pathway.parquet` | 89 | Ensembl gene to Reactome pathway mappings |
| `drug_gene.parquet` | 7 | ChEMBL mechanism-derived drug-target links |
| `gene_go.parquet` | 120 | GOA annotations obtained through QuickGO |
| `pathway_pathway.parquet` | 48 | Reactome parent-child pathway links |

`manifest.json` records source versions, retrieval rules, table schemas, row
counts, byte sizes, and SHA-256 checksums. The refresh implementation is
`scripts/refresh_sample_dataset.py`.

## Provenance

| Source | Use | Snapshot/version | Source terms |
| --- | --- | --- | --- |
| Open Targets Platform | Disease name and disease-gene associations | Live GraphQL snapshot | CC0 1.0 |
| STRING | Human protein-interaction neighborhood | 12.0 | CC BY 4.0 |
| Ensembl REST | Gene-symbol to Ensembl mapping | Live REST snapshot | CC BY 4.0 |
| Reactome | Pathway membership and hierarchy | 96 | CC0 for database data |
| UniProt | Ensembl-to-UniProt mapping | 2026_02 | CC BY 4.0 |
| GOA through QuickGO | Gene Ontology annotations and terms | Live API snapshot | CC BY 4.0 |
| ChEMBL | Drug-target mechanisms and molecule metadata | ChEMBL 37, 2026-05-01 | CC BY-SA 3.0 |

Source URLs and exact release strings are retained in `manifest.json`. No
DisGeNET or proprietary records are included.

## Transformations

1. Open Targets associations were sorted by the platform association score. The
   top 24 were retained, plus six prespecified OA sanity-check genes when
   available: `MMP13`, `ADAMTS5`, `COL2A1`, `FRZB`, `SOX9`, and `WNT5A`.
2. STRING was queried for Homo sapiens with a required score of 700 and up to 30
   added network neighbors. Only genes resolvable to versionless Ensembl gene
   IDs were retained.
3. At most two Reactome pathways per gene were selected, preferring labels
   related to cartilage, extracellular matrix, collagen, OA, TGF/BMP, or Wnt.
4. At most two non-`NOT` GOA annotations per mapped UniProt accession were
   retained, with evidence codes and references.
5. At most two ChEMBL mechanisms were retained for each of four selected
   single-protein targets.
6. Tables were deduplicated, lexicographically ordered, and written as Parquet.

The Open Targets score is a source-specific evidence aggregation score. It is
not interpreted here as causality, therapeutic efficacy, or clinical validity.

## License And Redistribution

The sample preserves source attribution and source-specific terms. The
repository's Apache-2.0 code license does not relicense third-party data.
ChEMBL-derived content remains under CC BY-SA 3.0; CC BY records retain their
attribution requirements; CC0 records remain CC0.

Redistribution is justified because every included source explicitly permits
redistribution under the terms listed above, the extraction is small and
attributed, and no source with incompatible or proprietary redistribution terms
is used. Downstream redistributors must preserve this dataset card,
`manifest.json`, and applicable attribution/share-alike notices.

## Intended Use

- End-to-end pipeline and packaging tests.
- Tutorials and local development without multi-gigabyte downloads.
- Validation of schema, identifier, graph, split, training, and reporting code.

## Limitations

- One disease node and 30 positive disease-gene links are insufficient for
  scientific model comparison.
- Source APIs are living resources; regenerating later can produce a different
  snapshot even with identical selection logic.
- The sample is intentionally enriched around one STRING neighborhood and is
  not representative of the human interactome.
- Selected pathways, GO terms, drugs, and genes are incomplete.
- Absence of an edge means "not included in this sample", not a biological
  negative.
- The pipeline is transductive for node identities and non-target relations.
  See `docs/leakage-prevention.md` for the exact guarantees and remaining risks.
