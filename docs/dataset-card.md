# Dataset Card

## Scope

The repository currently distributes only the small OA-oriented dataset under
`data/sample/`. No production-scale graph snapshot is included.

The primary entity is knee osteoarthritis (`EFO_0004616`). Node families are
disease, gene, drug, pathway, and GO term. Relations cover disease-gene
association, PPI, pathway membership and hierarchy, drug targeting, and GO
annotation.

## Provenance

The machine-readable source record is `data/sample/manifest.json`. It records
release identifiers, retrieval date, URLs, transformations, row counts, and
SHA-256 checksums for every redistributed Parquet file.

Sources include Open Targets, STRING, Reactome, ChEMBL, UniProt, Ensembl, and
GOA/QuickGO. DisGeNET is excluded from the redistributed dataset.

## Licensing

- Open Targets and Reactome annotation and derived interaction data: CC0.
- STRING, UniProt, Ensembl, and GO/GOA data: CC-BY-4.0.
- ChEMBL data: CC-BY-SA-3.0.

ChEMBL-derived redistribution can trigger attribution and share-alike
obligations. Users remain responsible for reviewing current upstream terms.
Reactome software, database dumps, pathway illustrations, and branding have
separate terms and are not redistributed in this sample.

## Intended Use

The sample supports integration tests, deterministic pipeline execution,
method development, and training materials. It is not intended for estimating
clinical efficacy, target validity, or comprehensive OA biology.

## Limitations

Selection is OA-oriented and incomplete. Literature popularity, source
curation, identifier mapping, and fixed-size expansion introduce bias. Missing
associations are unlabeled rather than confirmed negatives.

Detailed sample construction is documented in
[`dataset-card-sample.md`](dataset-card-sample.md).
