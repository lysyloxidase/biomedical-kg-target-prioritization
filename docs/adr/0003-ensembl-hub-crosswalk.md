# ADR 0003: Ensembl-Gene Hub Crosswalk

## Status

Accepted.

## Decision

Normalize source gene and protein identifiers to Ensembl gene IDs (`ENSG...`) as
the central hub.

## Consequences

The graph uses stable business keys across Open Targets, STRING, Reactome,
ChEMBL, UniProt, and GOA. The build must write unmapped-ID logs and fail the
Phase 1 gate if seed-gene coverage is 90% or lower.
