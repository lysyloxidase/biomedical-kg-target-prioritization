# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog 1.1.0, and this project adheres to
Semantic Versioning.

## [1.0.0] - 2026-06-11

### Added

- Phase 1 OA-centric data foundation: six-source policy, Ensembl-gene hub
  normalization, canonical edge assembly, graph statistics, Neo4j loader, ADRs,
  caveats, and license documentation.
- Phase 2 PyG `HeteroData` export with five node types, per-type features,
  node-index maps, leakage-free splits, negative sampling strategies, and
  split-leakage CI tests.
- Phase 3 OGB-style filtered evaluation metrics and seven non-graph baselines:
  popularity, logistic regression, matrix factorization, text embeddings,
  Node2Vec, centrality, and KGE baselines.
- Phase 4 HGT hero model, GraphSAGE and R-GCN ablations, encoder-decoder link
  prediction, multitask heads, leakage-free training, and multi-seed summaries.
- Phase 5 empirical ablations: no-KG vs KG, KG vs KG+text, homogeneous vs
  relational vs heterogeneous, and negative-sampling/layers/features design
  grid with paired significance scaffolding and markdown/LaTeX table assembly.
- Phase 6 interpretability: Captum/Integrated Gradients wrapper, HGT attention
  records, meta-path explanations, known-target and hypothesis-framed case
  studies, and explanatory subgraph figures.
- Phase 7 productionization: optional FastAPI target-ranking endpoint, Docker
  API image, full CI with lint/type/test/split-leakage/smoke-train/docker-build
  gates, README credibility tables, smoke-train CLI, and v1.0.0 documentation.

### Caveats

- Full OA benchmark result tables require generated artifacts under `reports/`.
  This source checkout keeps missing full-run numbers explicit rather than
  replacing them with synthetic metrics.
