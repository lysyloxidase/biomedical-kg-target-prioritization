# Refactor Roadmap

This roadmap starts from the verified state documented in
`docs/audit-current-state.md`. It deliberately postpones broad model work until
the repository has one executable, auditable data path.

## Cross-phase rules

Every phase should preserve these invariants:

1. A command either produces its declared artifacts or fails nonzero with an
   actionable message. Missing inputs must not be reported as successful work.
2. Every artifact records the code revision, configuration hash, input hashes,
   creation time, and schema version.
3. No topology-derived feature or negative candidate may use held-out target
   edges.
4. A named algorithm must implement that algorithm. Lightweight substitutes
   must have distinct names.
5. Missing optional dependencies or checkpoints must fail closed unless the
   user explicitly selects a documented smoke/fallback mode.
6. Synthetic smoke metrics must never be presented as biomedical results.
7. Tests must assert behavior and scientific invariants, not only file
   existence or documentation text.

## Phase 2: Sample dataset and real pipeline

### Objective

Create a small, redistributable, source-shaped dataset and make the data and
graph stages execute from a clean clone without manually prepared artifacts.

### Implementation

- Introduce a validated top-level configuration schema covering project,
  profiles, paths, source releases, graph limits, splits, features, training,
  and evaluation. Remove the current divergence between `config.yaml`,
  `sources.yaml`, and `params.yaml`, or define explicit ownership and validation
  for each file.
- Add a `sample` profile with small real records from all included sources:
  Open Targets, STRING, Reactome, ChEMBL, UniProt, and GOA. Preserve source
  shape, stable IDs, and license attribution. Do not invent biological labels
  to fill missing source records.
- Store a committed sample manifest with source name, release, retrieval date,
  canonical URL, license, checksum, byte size, raw row count, and citation.
  ChEMBL-derived sample content must retain CC-BY-SA attribution.
- Add source download/copy adapters with atomic writes, timeouts, retry policy,
  checksum verification, and an explicit offline mode. Network access must not
  be hidden inside normalization functions.
- Add one orchestrator that runs:
  `acquire -> verify -> normalize -> crosswalk -> graph -> statistics`.
- Write normalized per-source tables, the combined crosswalk, unmapped-ID logs,
  canonical node/edge tables, and a build manifest.
- Enforce the crosswalk coverage gate and graph-schema gate in the CLI.
- Make Neo4j an optional export after graph construction, not a required step in
  the core reproduction path.
- Replace the informational `kgtp data` and `kgtp graph` commands with explicit
  subcommands such as:

```text
kgtp data acquire --profile sample
kgtp data normalize --profile sample
kgtp graph build --profile sample
kgtp reproduce --profile sample
```

- Correct missing-value handling, duplicate-score selection, GO qualifier
  policy, ChEMBL target filtering, and STRING undirected-edge canonicalization.
- Add sample-source schema tests based on the exact committed fixtures.

### Primary files

- `src/kgtp/config.py`
- `src/kgtp/cli.py`
- `src/kgtp/data/*.py`
- new `src/kgtp/data/pipeline.py`
- new `src/kgtp/data/manifest.py`
- `configs/*.yaml`
- `Makefile`
- `dvc.yaml`
- sample fixture and manifest directories

### Acceptance criteria

- `kgtp reproduce-small` succeeds from a clean clone with no
  pre-existing generated files and no Neo4j service.
- Re-running it produces byte-identical normalized/processed tables or an
  explicitly documented deterministic Parquet equivalence.
- Corrupting a sample input causes checksum verification to fail nonzero.
- Every generated table is represented in the build manifest.
- The CLI reports real row counts, normalization yields, unmapped IDs, node
  counts, and edge counts.
- No full production benchmark claim is made from the sample graph.

### Phase 2 implementation status (2026-06-12)

Implemented in Phase 2:

- A committed seven-table OA sample snapshot with checksums, source releases,
  licenses, and a dataset card.
- Independently executable `prepare-sample`, `normalize`, `assemble`,
  `features`, `split`, `train-baselines`, `train-gnn`, `evaluate`, and `report`
  stages.
- `make reproduce-small`, which runs the real Parquet-to-PyG pipeline without
  Neo4j and writes deterministic artifacts under `artifacts/sample/`.
- Stage manifests with declared inputs, outputs, SHA-256 hashes, row/count
  metadata, fixed seeds, and nonzero failure on missing prerequisites.
- A complete integration test, checkpoint reload during evaluation, and
  byte-identical artifacts across two fixed-seed runs on the audited machine.

Still deferred:

- Production-scale acquisition, unified production configuration, crosswalk
  coverage reports, and unmapped-ID reports.
- Split-first feature construction and leakage-safe topology features, which
  remain Phase 3 work.
- The full set of corrected baselines and multi-model benchmark runs from
  Phases 4 and 5.

## Phase 3: Split-first architecture and leakage prevention

### Objective

Make edge partitioning precede every target-dependent topology feature and
construct model inputs only from the permitted training graph.

### Implementation

- Split canonical positive edge tables before building model features or
  `HeteroData`.
- Define the exact prediction tasks. If disease-gene is primary and drug-gene
  and gene-pathway are auxiliary, document and version that policy.
- Produce four distinct artifacts:
  full known-positive registry, split assignment table, train message graph,
  and supervision tables.
- Build train/validation/test `HeteroData` views from the split artifacts.
  Validation and test message graphs must not include validation or test target
  edges.
- Fit structural feature builders only on the train message graph. Apply the
  fitted feature schema to all nodes without consulting held-out edges.
- Compute degree, PageRank, pathway participant counts, GO information content,
  and any future topology feature from the allowed graph view.
- Decide whether non-target relations are frozen as full transductive context
  or temporally/split restricted. Record the decision per relation.
- Keep full known positives only for filtered evaluation and false-negative
  exclusion; never pass them into the encoder or feature builder.
- Persist split seeds, edge hashes, node maps, relation policy, and feature-fit
  graph hash.
- Add explicit inductive/transductive terminology to docs and artifact
  manifests.

### Tests

- Perturbing or deleting a validation/test edge must not change training
  features, train message edges, or train negatives.
- Every held-out edge and reverse edge must be absent from every message graph
  that is not allowed to contain it.
- Split partitions must be disjoint and reconstruct the original positive set.
- Split behavior must be deterministic for a fixed seed and change for a
  different seed.
- Isolated nodes and single-source disease cases must be covered.

### Acceptance criteria

- The diagnostic that currently changes a gene feature after removing a test
  edge must report no change in the training feature artifact.
- Coverage includes configuration, split orchestration, and HeteroData
  construction; these modules are no longer omitted.
- A machine-readable leakage report is emitted with the split artifacts.

### Phase 3 implementation status (2026-06-12)

Implemented:

- Canonical disease-gene splitting before PyG construction or feature fitting.
- Separate full-reference, train-message, assignment, known-positive, and
  train/validation/test supervision artifacts.
- A serializable `TrainGraphFeatureTransformer` with explicit
  `fit(train_graph)` and edge-independent `transform(nodes)` behavior.
- Train-only degree, PageRank, GO vocabulary, pathway, and GO information
  features shared identically by all model views.
- Restricted evaluation references containing known positives and candidate
  node IDs rather than a full encoder graph.
- Semantic graph, partition, transformer-state, and node-index hashes.
- Hard validation for duplicate edges, overlap, incomplete reconstruction,
  invalid negatives, reverse-edge contamination, and hash mismatch.
- Perturbation, corruption, reverse-edge, supervision-isolation, and seed tests
  in `tests/unit/test_leakage_prevention.py`.

Still deferred:

- Temporal splitting of non-target context.
- Correct degree-matched and hard-negative strategies.
- Hyperparameter-search orchestration and test-set access controls.
- Inductive evaluation with unseen nodes.

## Phase 4: Negative sampling and baselines

**Status:** implemented for the executable sample pipeline. Production-scale,
multi-seed tuning remains deferred.

### Objective

Implement genuinely distinct negative strategies and executable, correctly
named baseline benchmarks.

### Negative sampling

- Introduce a sampler protocol with explicit `fit(train_graph)` and
  `sample(query, count, forbidden)` behavior.
- Keep a uniform random sampler as the reference.
- Implement degree-matched sampling against a documented target-degree bin or
  nearest-degree distribution derived only from the train graph.
- Implement hard negatives using a defined train-only signal, for example:
  two-hop PPI/pathway proximity, text-nearest candidates, or model-mined
  candidates from a previous training checkpoint.
- Define behavior when a hard pool is too small and record fallback counts
  instead of silently becoming random.
- Persist sampled negatives and diagnostics: overlap checks, source/target
  degree distributions, duplicate rate, fallback rate, and candidate-pool size.
- Separate training-negative strategy from evaluation candidate protocol.

### Baselines

- Rename the current adjacency-SVD model to `AdjacencySVDBaseline`.
- Add a real Node2Vec/DeepWalk implementation with random walks and skip-gram,
  or remove the Node2Vec claim.
- Make hash text vectors an explicit `HashTextBaseline`.
- Make the PubMedBERT/sentence-transformer baseline require its model,
  dependency, revision, cache manifest, and embedding dimension. Missing models
  must fail nonzero.
- Implement TransE, DistMult, ComplEx, and RotatE with correct parameterization
  and losses, preferably through a pinned optional PyKEEN dependency. Delete the
  zero-valued PyKEEN scorer.
- Ensure no-KG logistic regression receives explicit, split-safe features and
  never silently hashes missing node features.
- Wire `kgtp baselines run` to train/evaluate every selected baseline across
  configured seeds and write result artifacts.

### Tests

- Fixed-seed random, degree-matched, and hard samplers must not be identical on
  a fixture designed to distinguish them.
- Degree-matched output must satisfy a quantitative distribution tolerance.
- Hard negatives must satisfy the configured hardness criterion.
- Formula-level tests must compare KGE scores with hand-computed examples.
- Node2Vec tests must verify walks/training, not merely embedding shape.
- Text-model absence must raise a clear error.

### Acceptance criteria

- One command produces all baseline reports from Phase 3 split artifacts.
- Every report records algorithm, dependency/model revision, feature manifest,
  split hash, seed, runtime, and metrics.
- README baseline names exactly match executed implementations.

## Phase 5: GNN training and evaluation

**Status:** implemented for the five-seed sample experiment. Production-scale
mini-batching, tuning, and scientific validation remain deferred.

### Objective

Run HGT, GraphSAGE, and R-GCN under one leakage-safe, multi-seed protocol and
produce scientifically interpretable evaluation artifacts.

### Implementation

- Define model-specific configuration schemas and checkpoint manifests.
- Align GraphSAGE naming with implementation: use real `to_hetero` if claimed,
  or document the relation-wise `HeteroConv` model.
- Use a mathematically documented R-GCN implementation. If basis decomposition
  is configurable, implement it; otherwise remove `num_bases`.
- Add normalization, dropout, residual behavior, and task-loss weighting only
  where justified and covered by configuration.
- Replace the one-seed CLI with a multi-seed runner using the same split and
  feature artifacts as baselines.
- Select checkpoints using a declared validation metric. If AUPRC is primary,
  do not silently optimize `AUPRC + MRR`.
- Enable deterministic algorithms where supported and record exceptions,
  hardware, device, and library versions.
- Define one evaluation protocol:
  full-candidate filtered ranking when feasible, or a fixed, persisted sampled
  candidate set. State whether head, tail, or both corruptions are evaluated.
- Keep AUROC secondary and report class prevalence and negative ratio with every
  binary metric.
- Add calibration metrics only if downstream probability interpretation is
  intended.
- Use paired comparisons on identical seeds/splits and correct for multiple
  comparisons in large ablation grids.
- Enforce the popularity floor as a reported gate, not a standalone helper.

### Artifacts

- per-seed checkpoint and checkpoint manifest;
- loss and validation history;
- per-query ranks and candidate-set hashes;
- scalar metric report;
- mean, standard deviation, confidence interval, and paired-comparison report;
- consolidated benchmark table generated only from saved reports.

### Tests and gates

- Assert a meaningful optimization property on a learnable fixture; do not use
  a tautological `min(losses) <= losses[0]` check.
- Test checkpoint reload and exact score reproduction.
- Test model/data manifest incompatibility failure.
- Test CPU training for all three model families.
- Add a bounded sample end-to-end training gate to CI.

### Acceptance criteria

- Baselines and all GNNs run from the same Phase 3 artifacts across at least five
  configured seeds.
- Result tables are generated from real saved reports with no hand-entered
  values.
- No claim of superiority is made unless supported by the persisted metrics and
  comparison protocol.

## Phase 6: Explainability and API

**Status:** implemented for validated sample artifacts. Production deployment,
latency controls, and stronger interpretability validation remain deferred.

### Objective

Generate explanations and serve predictions only from a validated trained
checkpoint.

### Explainability

- Require checkpoint, model manifest, graph/split manifest, and compatible node
  maps. The normal explain command must fail if any are missing.
- Generate rankings from model scores and explicitly exclude or label known
  train positives.
- Use native model attention only if it is actually exposed and validated.
  Rename endpoint-conditioned heuristics as proxy visualizations and do not
  label them HGT attention.
- Distinguish learned attribution, graph-path evidence, and curated biology in
  the output schema.
- Add faithfulness checks such as deletion/insertion or score change after
  masking top-ranked features/edges.
- Prevent fallback index ordering from becoming a novel-target ranking.
- Keep known-target and novel-hypothesis narratives, but derive them from real
  checkpoint output.

### API

- Return `503 Service Unavailable` from `/predict` when no validated checkpoint
  is loaded.
- Make `/health` distinguish process health, artifact readiness, and model
  readiness.
- Validate checkpoint architecture, split hash, feature schema, and node-map
  hash at startup.
- Prefer safe tensor/state formats and avoid loading untrusted pickle artifacts.
- Bound `top_k`, explanation count, graph payload size, and request time.
- Cache embeddings and optionally explanations instead of recomputing the
  encoder and integrated gradients per returned gene.
- Include model version, dataset version, split ID, and disclaimer in every
  prediction response.
- Separate smoke/demo mode from production mode with an explicit environment
  flag and visible response field.

### Acceptance criteria

- A fresh clone without a checkpoint cannot return target rankings.
- A deliberately mismatched checkpoint is rejected at startup.
- API tests cover ready, not-ready, mismatch, malformed artifact, and bounded
  request cases.
- Case studies can be regenerated from a named result/checkpoint manifest.

### Phase 6 implementation status (2026-06-12)

Implemented:

- One fail-closed artifact loader shared by the explain CLI and FastAPI.
- Compatibility validation for checkpoint configuration, dataset and graph
  hashes, split metadata, train-message graph, feature transformer, node-index
  map, validation metrics, and supervision/message-edge separation.
- Required-argument `kgtp explain` execution from a trained checkpoint only.
- Full-candidate model ranking and structured evidence cards for known,
  held-out, sampled-unlabeled, and novel-hypothesis cases.
- Integrated Gradients feature attribution and model-score edge occlusion
  without heuristic fallback.
- Explicitly non-causal attribution terminology; endpoint-conditioned topology
  summaries are no longer labeled as HGT attention.
- HTTP 503 for missing, corrupt, untrained, smoke-only, or incompatible runtime
  artifacts.
- Explicit `KGTP_DEMO_MODE=true` behavior with prominent non-scientific
  warnings.
- Integration tests for ready, missing, corrupt, graph mismatch, feature
  mismatch, demo, unknown disease, and invalid `top_k` cases.

Remaining:

- Checkpoints still use PyTorch serialization and must be treated as trusted
  local artifacts.
- Integrated Gradients baselines and edge-occlusion neighborhoods are simple
  research defaults, not validated biological explanation methods.
- API embedding and explanation caching, request timeouts, authentication, and
  production observability remain Phase 7 work.

## Phase 7: Manifests, CI, security and documentation

**Status:** implemented for the sample pipeline and pull-request CI. Independent
release verification and full-data operations remain Phase 8 work.

### Objective

Make provenance, licensing, CI, packaging, and public claims match the actual
pipeline.

### Provenance and licensing

- Correct Reactome data licensing to CC0 while separately documenting artwork,
  software, and dump exceptions where relevant.
- Replace `LICENSE` with the canonical Apache License 2.0 text.
- Add `CITATION.cff`, `SECURITY.md`, data and model cards, and attribution
  notices for redistributed sample data.
- Record GO release date/DOI and all source retrieval/checksum metadata.
- Add a generated data-provenance table to documentation from manifests.
- Document ChEMBL share-alike implications for redistributed derived datasets.

### CI

- Add clean-clone sample reproduction and artifact-schema validation.
- Remove critical-module coverage omissions and use risk-based coverage
  thresholds.
- Run leakage invariants, sampler semantics, checkpoint reload, API readiness,
  docs links, and license-manifest checks.
- Pin GitHub Actions to maintained major versions or commit SHAs and resolve the
  Node.js runtime deprecation warnings.
- Add dependency review, secret scanning, static security checks, and an SBOM
  or vulnerability scan for the container.
- Keep full production data/model jobs scheduled or manual because of cost, but
  publish their manifest and status separately from PR smoke CI.

### Docker

- Pin `uv` installation, use a non-root runtime user, add a healthcheck, and use
  a smaller runtime stage.
- Do not bake mutable data or checkpoints into the image without a manifest.
- Remove default demo predictions from the production image.
- Replace default Neo4j credentials in examples with required environment
  configuration.

### Documentation

- Rewrite README claims from generated manifests and result reports.
- Remove "first" unless supported by a documented literature review.
- Describe the evaluation protocol precisely rather than relying on
  "OGB-style".
- Use the correct names for adjacency SVD, real Node2Vec, hash text, and KGE
  implementations.
- Keep result tables pending until Phase 5 artifacts exist.
- Align package, changelog, API, image, and documentation versions.

### Acceptance criteria

- CI verifies the complete sample path, not only synthetic unit tests.
- Documentation build fails on missing generated references or broken links.
- License/provenance checks cover every redistributed sample file.
- Container runs as non-root and refuses prediction without validated artifacts.

### Phase 7 implementation status (2026-06-12)

Implemented:

- Run-level manifests with Git state, environment, lock/config hashes, source
  versions and licenses, file hashes, graph/split/feature/checkpoint hashes,
  seeds, command, and terminal status.
- Compatibility validation before sample training and evaluation, plus
  completed-run validation for explanations and API serving.
- Separate CI gates for quality, unit tests, sample integration, leakage,
  determinism, sampled-unlabeled semantics, API safety, coverage, smoke
  training, dependency review, security scans, and Docker build.
- Risk-based coverage thresholds for split, feature, evaluation, artifact, API,
  and pipeline modules.
- Environment-only Neo4j credentials, `.env.example`, non-root container
  execution, and readiness tied to validated artifacts.
- Package, API, model artifact, changelog, and documentation version alignment
  at pre-1.0 version `0.2.0`.
- Dataset, model, evaluation, reproducibility, biomedical-limitations, and API
  safety documentation.

Remaining:

- GitHub-hosted CI status and remote dependency-review results require a pushed
  commit and are not claimed by this local phase.
- The container is built locally in Phase 7 but clean-room execution, SBOM
  publication, signing, and retained release evidence remain Phase 8.
- Full-data manifests and scientific benchmark results do not exist.

## Phase 8: Final verification

### Objective

Perform an independent clean-room verification and freeze the release evidence.

### Verification procedure

1. Clone the repository into an empty directory at the release commit.
2. Run the locked environment sync.
3. Run lint, format, mypy, pyright, unit, integration, leakage, and coverage
   checks.
4. Reproduce the sample data, graph, splits, features, baselines, GNN smoke
   training, evaluation, explanation, and API readiness flow.
5. Re-run the sample pipeline and compare artifact hashes.
6. Build and run the Docker image, including health/readiness and prediction
   failure without artifacts.
7. Verify the latest GitHub Actions run and retained artifacts.
8. Validate manifests, source URLs, checksums, licenses, citations, model cards,
   and result-table provenance.
9. Run the full benchmark only where source terms, compute, and storage permit.
10. Publish a final verification report listing every command, revision,
    environment, artifact hash, and unresolved limitation.

### Required command baseline

```text
git status
python --version
uv sync --frozen --extra dev --no-editable
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright src tests
uv run pytest -q
uv run pytest --cov=kgtp --cov-report=term-missing
kgtp reproduce --profile sample
docker build -f docker/Dockerfile -t kgtp-final .
```

### Final acceptance criteria

- The worktree remains clean after verification except for documented,
  ignored generated artifacts.
- Sample reproduction succeeds twice with matching manifests and hashes.
- No critical leakage, provenance, licensing, checkpoint-readiness, or
  algorithm-label mismatch remains open.
- All public result values trace to immutable per-seed reports.
- All README claims are classified as demonstrated, pending, or explicitly out
  of scope.
- Blocked production checks are documented without fabricated substitutes.

## Phase dependencies

```text
Phase 2 data pipeline
    -> Phase 3 split-first artifacts
        -> Phase 4 negatives and baselines
        -> Phase 5 GNN training and evaluation
            -> Phase 6 explainability and API
                -> Phase 7 production hardening and documentation
                    -> Phase 8 final verification
```

Phase 4 and Phase 5 may share evaluation infrastructure after Phase 3, but
neither should start benchmark claims before the Phase 2 and Phase 3 artifact
contracts are stable.
