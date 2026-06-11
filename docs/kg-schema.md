# Knowledge Graph Schema

## Node Types

| Label | Business Key | Examples |
| --- | --- | --- |
| Disease | EFO ID | `EFO_0004616` |
| Gene | Ensembl gene ID | `ENSG00000105829` |
| Pathway | Reactome stable ID | `R-HSA-1474244` |
| Drug | ChEMBL molecule ID | `CHEMBL2107357` |
| GOTerm | GO ID | `GO:0001501` |

## Edge Types

| Triplet | Required | Source |
| --- | --- | --- |
| `(Disease, associated_with, Gene)` | Yes | Open Targets |
| `(Gene, interacts, Gene)` | Yes | STRING |
| `(Gene, participates_in, Pathway)` | Yes | Reactome |
| `(Drug, targets, Gene)` | Yes | ChEMBL |
| `(Gene, annotated_with, GOTerm)` | Yes | GOA |
| `(Pathway, parent_of, Pathway)` | Optional | Reactome |

All node and edge tables include stable IDs, source attribution, and enough
properties to reproduce filtering decisions.
