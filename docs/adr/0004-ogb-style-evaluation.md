# ADR 0004: OGB-Style Evaluation

## Status

Accepted.

## Decision

Use OGB-style train/validation/test splits and negative sampling for downstream
link prediction.

## Consequences

Phase 1 preserves enough edge provenance to split only valid positive
disease-gene links in later phases. The exact split implementation belongs to
Phase 3, but the data foundation records the decision now.
