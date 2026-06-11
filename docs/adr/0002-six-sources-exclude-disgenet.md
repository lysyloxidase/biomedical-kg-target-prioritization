# ADR 0002: Six Open Sources And DisGeNET Exclusion

## Status

Accepted.

## Decision

Ingest Open Targets, STRING, Reactome, ChEMBL, UniProt, and GOA. Exclude
DisGeNET because its current redistribution model is not compatible with the
intended benchmark distribution.

## Consequences

Open Targets genetic-association evidence is used as the open substitute for
disease-gene genetics. Tests and documentation should make the exclusion
explicit.
