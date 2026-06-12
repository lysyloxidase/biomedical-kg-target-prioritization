# Data Sources

| Source | Version | Role | License | Notes |
| --- | --- | --- | --- | --- |
| Open Targets | 25.06 | Disease-gene seeds, genetics, MoA, tractability | CC0-1.0 | Disease anchor `EFO_0004616`; OA count read live from GraphQL |
| STRING | 12.0 | High-confidence human PPI expansion | CC-BY-4.0 | Species 9606; combined score >= 700 |
| Reactome | v90 | Gene-pathway and optional pathway hierarchy | CC0-1.0 for annotation and derived interaction data | Software, dumps, illustrations, and branding have separate terms |
| ChEMBL | 35 | Drug-target mechanism edges | CC-BY-SA-3.0 | Share-alike attribution must be preserved |
| UniProt | 2025_03 | UniProt-Ensembl crosswalk and protein metadata | CC-BY-4.0 | Used for ChEMBL target normalization |
| Gene Ontology / GOA | 2025-05 | Gene-GO annotations | CC-BY-4.0 | Human GAF bulk |
| DisGeNET | Excluded | Not ingested | Restricted/freemium | Open Targets genetic association is the open substitute |

## Required Audit Fields

Every source snapshot should record retrieval date, release/version, URL or API
endpoint, file checksum, row count, normalized row count, license, and the
normalization yield.

Official license references:

- Reactome: <https://reactome.org/license>
- ChEMBL: <https://www.ebi.ac.uk/chembl/>
- Gene Ontology: <https://geneontology.org/docs/go-citation-policy/>
