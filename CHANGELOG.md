# Changelog

All notable changes are documented here using Keep a Changelog conventions.
The project uses pre-1.0 semantic versioning while the full biomedical
benchmark remains incomplete.

## [Unreleased]

### Planned

- Independent clean-room verification and release evidence.
- Full-data acquisition and production-scale benchmark execution.

## [0.2.0] - 2026-06-12

### Added

- Redistributable OA sample dataset and executable `make reproduce-small`
  pipeline.
- Split-first train-message graph and train-fitted structural features.
- Distinct random, degree-matched, and train-only hard sampled-unlabeled
  strategies.
- Correctly named adjacency SVD, biased-walk Node2Vec, text, matrix
  factorization, MLP, and KGE baseline implementations.
- HGT, heterogeneous and homogeneous GraphSAGE, and R-GCN multi-seed runner.
- Full-candidate and sampled-unlabeled evaluation with ranking, calibration,
  enrichment, uncertainty summaries, and paired comparisons.
- Trained-artifact-only explainability and fail-closed API behavior.
- Run-level provenance manifests, risk-based coverage gates, security scans,
  container health checks, and non-root container execution.

### Changed

- API, package, and documentation versions are aligned at `0.2.0`.
- Claims are limited to the executable sample pipeline; no full-scale
  scientific benchmark result is claimed.
- Model explanations are described as non-causal attributions.
- Neo4j credentials must be supplied through environment variables.

### Security

- Removed hard-coded default passwords.
- API readiness requires validated checkpoint, graph, feature, split,
  validation, and run manifests.
- Added dependency and static security scans to CI.

### Known limitations

- Sample metrics validate software execution only.
- Full-scale data acquisition, tuning, external validation, and prospective
  biological validation have not been completed.
- Optional Sentence Transformer and PubMedBERT arms are unavailable unless
  their explicit dependencies and model weights are installed.

## [0.1.0] - 2026-06-11

### Added

- Initial research scaffold, data-source policy, graph schema, model
  prototypes, tests, and documentation.
