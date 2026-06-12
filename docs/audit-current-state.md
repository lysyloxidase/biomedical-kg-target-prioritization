# Current-State Repository Audit

Audit date: 2026-06-12

> This document records the Phase 1 state at revision
> `4982c60e08ef8162f70cb0f614e901515bf81215`, before the Phase 2 sample
> pipeline was implemented. See `docs/refactor-roadmap.md` for the dated Phase 2
> implementation status.

Repository: `lysyloxidase/biomedical-kg-target-prioritization`

Audited revision: `4982c60e08ef8162f70cb0f614e901515bf81215`
(`main`, synchronized with `origin/main` at the start and end of the audit)

## Scope and method

This audit inspected the repository tree, documentation, Makefile, DVC file,
CLI, configuration, source ingestion helpers, graph assembly, identifier
normalization, PyG export, split and negative-sampling logic, baselines, GNN
models, training and evaluation, explainability, API, tests, Docker, GitHub
Actions, dependency configuration, licenses, and data-provenance claims.

The audit combined:

- static inspection of all Python modules and tests;
- execution of the requested lint, type, test, coverage, and Docker commands;
- direct execution of the important CLI stages on a fresh checkout;
- focused diagnostic scripts for structural-feature leakage, negative samplers,
  KGE behavior, and text-embedding fallback;
- inspection of the latest public GitHub Actions runs;
- verification of source-license claims against official source websites.

No production dataset, full benchmark report, or model checkpoint was present.
No scientific performance claim was therefore evaluated as if it were a real
benchmark result.

## Executive conclusion

The repository is a well-formatted and well-tested research scaffold, not a
working end-to-end biomedical benchmark. It contains useful normalization
functions, graph schemas, PyG model implementations, metrics, and synthetic
tests. However, the top-level pipeline does not acquire or process source data,
does not run baseline benchmarks or ablations, and cannot reproduce the README
workflow from a clean clone.

The most serious scientific issue is split ordering. Structural features are
computed from the complete graph before positive target edges are assigned to
train, validation, and test. Degree and PageRank therefore encode held-out
edges. The existing split tests correctly check direct edge overlap but do not
test feature leakage.

The baseline and explainability labels also exceed their implementations:
Node2Vec is adjacency SVD, the default text baseline is hashing, the NumPy KGE
training objective is DistMult for every named model, and the explainability CLI
uses a newly initialized model plus an index-based artificial ranking. The API
can similarly return rankings from an untrained model while reporting health
status `ok`.

## Repository state and artifacts

- Git worktree was clean before the audit.
- Tracked data content consists only of `data/README.md` and `.gitkeep` files.
- There is no tracked or local `reports/` directory.
- There are no full-run result JSON files, ablation reports, figures, model
  checkpoints, processed graph tables, HeteroData exports, or split artifacts.
- `uv.lock` exists and `uv sync --frozen --extra dev --no-editable` succeeds.
- The project metadata version is `0.1.0`, while `CHANGELOG.md` and the FastAPI
  application identify version `1.0.0`.

## Requested issue verification

| Issue to verify | Verdict | Evidence |
| --- | --- | --- |
| `make reproduce` is not a real end-to-end pipeline | Verified | `Makefile:43` chains commands, but `data` and `baselines` are informational, `graph` and `explain` exit successfully when artifacts are absent, `ablate` requires pre-existing reports, and `neo4j` currently fails while loading configuration. |
| Data commands only inspect configuration instead of downloading or processing data | Verified | `kgtp data` at `src/kgtp/cli.py:47` prints source configuration and optionally fetches one live count. `kgtp graph` at line 70 only validates existing processed tables. No source downloader or orchestration entry point exists. |
| Baseline commands do not execute benchmark models | Verified | `kgtp baselines` at `src/kgtp/cli.py:152` only prints seven expected output paths. |
| Ablation commands depend on previously generated reports | Verified | `kgtp ablate` at `src/kgtp/cli.py:202` reads four JSON reports and exits 1 when they are absent. It does not invoke the ablation evaluators. |
| Structural features are computed before target-edge splitting | Verified | `build_heterodata()` calls `_gene_features()` while all canonical edge tables are present (`src/kgtp/hetero/build_heterodata.py:79`). Splitting happens later in `kgtp splits`. |
| Degree and PageRank can leak test-edge information | Verified | `_gene_structural_features()` counts every endpoint and computes PageRank from all edge tables (`build_heterodata.py:281-299`). A diagnostic held out `EFO_0004617 -> ENSG00000100005`; removing only that test edge changed its structural feature from `[6.0, 0.0582165]` to `[5.0, 0.0508111]`. |
| Hard and degree-matched negative sampling behave like random sampling | Verified for the production split path | `splits.py` passes neither degree tensors nor hard candidates to `sample_negative_edges()` (`splits.py:276`). Diagnostic runs produced exactly identical test `edge_label_index` tensors for `random`, `degree_matched`, and `hard` on all three predicted relations. |
| The model called Node2Vec is actually adjacency SVD | Verified | `src/kgtp/baselines/node2vec.py:28-40` constructs an adjacency matrix and calls `np.linalg.svd`; it performs no random walks or skip-gram training. |
| Text embeddings silently fall back to hash vectors | Verified | `text_embeddings.py:65-70` hashes text when no model is configured or `sentence_transformers` is unavailable. The dependency is not declared. A diagnostic request for a nonexistent model returned a 64-dimensional hash vector without an error. |
| KGE implementations do not mathematically correspond to their names | Verified | `_sgd_step()` always optimizes the DistMult score (`kge.py:111-123`). ComplEx has no complex embeddings, RotatE has no phase/complex rotation and scores identically to TransE, and `_score_pykeen()` always returns `0.0`. Diagnostics confirmed identical trained embeddings for all four names and identical TransE/RotatE scores. |
| Explainability uses an untrained model or artificial ranking | Verified | `kgtp explain` builds a fresh HGT (`src/kgtp/cli.py:249`) and ranks genes by `gene_count - gene_idx` (`cli.py:262`). Attention and edge masks are explicitly endpoint-conditioned proxies (`explain/attention.py:52-61`, `explain/explainer.py:424`). |
| API can serve predictions from an untrained model | Verified | `get_service()` defaults to `model_source = "untrained-smoke-model"` (`api/app.py:133`) and continues if no checkpoint exists or loading fails. `/health` still returns `"status": "ok"` and `/predict` remains enabled. |
| Critical modules are excluded from coverage | Verified | `pyproject.toml:87-102` omits the CLI, configuration, most ingestion modules, graph builder, HeteroData builder, Neo4j loader, logging, and seeds. Reported 87% coverage therefore excludes much of the end-to-end risk. |
| Full benchmark results and artifacts are absent | Verified | Git and filesystem inspection found no `reports/`, processed data, checkpoints, manifests, or benchmark result files. README result cells are marked pending. |
| README claims exceed what the implementation demonstrates | Verified | Claims of an open reproducible benchmark, leakage-free splits, empirical ablations, PubMedBERT, Node2Vec, named KGEs, `to_hetero` GraphSAGE, interpretability, and end-to-end reproduction are not demonstrated by the current executable pipeline. |

## Additional high-priority findings

### 1. Neo4j CLI is broken before it reads graph data

`configs/config.yaml` contains a top-level `project` section. `Settings` does not
declare that field and Pydantic rejects it as extra input. `uv run kgtp neo4j`
failed with:

```text
ValidationError: 1 validation error for Settings
project
  Extra inputs are not permitted
```

The same configuration also contains fields not represented in
`GraphSettings`, including `canonical_edge_types`. Configuration ownership is
split between `configs/config.yaml`, `configs/sources.yaml`, and `params.yaml`
without one validated schema.

### 2. Feature-mode names are not implemented as claimed

`_gene_features()` recognizes only `none` and `go` specially. Every other mode
uses structural features and optionally appends an existing vector column.
Consequences:

- the ablation value `one-hot` does not select one-hot features;
- `ESM` does not compute or require ESM embeddings;
- `text` does not compute or require text embeddings;
- a missing requested vector silently produces structural-only features;
- the KG, KG+text, ESM, and one-hot arms can be behaviorally identical.

Drug features similarly fall back from Morgan fingerprints to ID/SMILES hashing
because RDKit is not a declared dependency.

### 3. Evaluation is only loosely "OGB-style"

The implementation has deterministic filtered tail ranking and useful metric
tests, but it does not use an OGB evaluator or OGB-provided negative sets. It:

- corrupts tails only;
- samples up to `negatives_per_positive` candidates;
- resamples evaluation negatives inside `evaluate_binary_and_ranking()`;
- does not use the split's hard or degree-matched negatives for evaluation;
- selects checkpoints by `AUPRC + filtered_MRR`, despite documenting AUPRC as
  the primary metric.

This is a project-specific sampled filtered-ranking protocol, not a verified
reproduction of an OGB benchmark protocol.

### 4. Reproducibility controls are incomplete

`set_deterministic()` seeds Python, NumPy, and Torch but explicitly calls
`torch.use_deterministic_algorithms(False)`. The repository does not record
hardware, Torch/PyG runtime versions, split hash, feature hash, dataset manifest,
or checkpoint architecture metadata in one artifact. Multi-seed functions exist
but the CLI trains one seed only.

### 5. Ingestion helpers are not yet a data pipeline

The source modules normalize already available dataframes or files. They do not:

- download pinned releases;
- verify URLs, checksums, sizes, or media types;
- write retrieval and normalization manifests;
- orchestrate crosswalk construction;
- enforce the configured graph-size and crosswalk gates;
- assemble and persist the six-source graph from the CLI.

Important source-specific policies are also absent, including GO `NOT`
qualifier handling, evidence-code policy, STRING undirected-edge
canonicalization, and explicit ChEMBL target-type filtering.

### 6. PPI message passing is directed

Canonical STRING interactions are conceptually undirected, and structural
features treat them as undirected. The HeteroData reverse-edge map does not add
the reverse of `("gene", "interacts", "gene")`. Unless both orientations happen
to be present in the input table, GNN message passing sees a directed PPI graph.

### 7. Explainability labels overstate the generated evidence

The code correctly labels novel targets as hypotheses, but:

- the CLI never loads a trained checkpoint;
- generated "attention" is a deterministic endpoint heuristic, not HGT
  attention;
- fallback edge importance is endpoint proximity, not learned attribution;
- fallback case-study candidates can be selected by index when no ranking is
  available;
- tests use randomly initialized toy models and verify artifact shape, not
  explanatory faithfulness.

### 8. API readiness and artifact safety are insufficient

The API:

- serves random rankings when a checkpoint is absent;
- reports healthy status even after checkpoint load failure;
- has no checkpoint/data compatibility manifest;
- uses `torch.load(..., weights_only=False)` for graph and split artifacts,
  which must be treated as trusted pickle input;
- computes explanations for every returned target synchronously;
- exposes all graph data without an explicit deployment or size policy.

### 9. Coverage and tests validate scaffolding more than behavior

The 41 tests are useful unit/smoke tests, but several assertions are weak:

- the loss-decrease assertion uses `min(losses) <= losses[0]`, which is true
  even when loss never improves;
- KGE tests check declared names, not model mathematics;
- negative-sampler tests check forbidden-edge exclusion, not strategy
  semantics;
- quality-gate tests check that claims are present in docs, not that the
  claimed pipeline exists;
- API tests intentionally accept the untrained fallback service.

### 10. Licensing and provenance need correction

Official source pages support the repository's Open Targets CC0, STRING
CC-BY-4.0, ChEMBL CC-BY-SA-3.0, UniProt CC-BY-4.0, and GO CC-BY-4.0 claims.
Reactome is incorrect: Reactome database data and derived data files are CC0,
not CC-BY-4.0.

The repository `LICENSE` file is a shortened rewrite of Apache-2.0 rather than
the canonical Apache License 2.0 text. This should be replaced with the standard
license text. `LICENSE-figures` is only a short notice and link.

No downloaded snapshot exists, so no actual data artifact currently has:

- retrieval date;
- source URL;
- release identifier and DOI where applicable;
- checksum;
- raw and normalized row counts;
- normalization yield;
- license and attribution text;
- transformation-code revision.

### 11. Docker and CI are smoke gates only

The local Docker build could not run because the Docker Desktop Linux daemon was
not running. This is an environment blocker, not evidence of a Dockerfile
failure.

The latest public CI run for commit `4982c60` completed successfully on
2026-06-11. Its three jobs were lint/type/test, smoke-train, and Docker build.
That verifies the synthetic suite and image build, not the source-data pipeline
or a benchmark run. GitHub also reports Node.js 20 deprecation warnings for
`actions/checkout@v4` and `astral-sh/setup-uv@v3`.

The Docker image installs an unpinned latest `uv`, runs as root, has no
healthcheck, and defaults to an API that may serve an untrained model.

## Commands and actual outcomes

### Requested commands

| Command | Outcome |
| --- | --- |
| `git status` | Passed. `main...origin/main`; clean worktree. |
| `python --version` | Passed. `Python 3.12.10`. |
| `uv sync --frozen --extra dev --no-editable` | First invocation exceeded the 120-second audit wrapper while `uv` continued installing. After the process exited, the exact command was rerun and passed: `Checked 108 packages in 6ms`. |
| `uv run ruff check .` | Passed: `All checks passed!` |
| `uv run ruff format --check .` | Passed: `69 files already formatted`. |
| `uv run mypy src` | Passed: `Success: no issues found in 58 source files`. |
| `uv run pyright src tests` | Passed: `0 errors, 0 warnings, 0 informations`. |
| `uv run pytest -q` | Passed: 41 tests, 3 warnings. |
| `uv run pytest --cov=kgtp --cov-report=term-missing` | Passed: 41 tests, 3 warnings, reported total 87% branch-aware coverage. |
| `docker build -f docker/Dockerfile -t kgtp-audit .` | Not executed by the daemon. Docker CLI failed to connect to `dockerDesktopLinuxEngine` because the named pipe did not exist. |

Test warnings:

- one Starlette/FastAPI `TestClient` deprecation warning concerning `httpx`;
- two TorchScript deprecation warnings recommending `torch.compile` or
  `torch.export`.

### Additional pipeline checks

| Command | Outcome |
| --- | --- |
| `make --version` | Failed: GNU Make is not installed on this Windows environment, so `make reproduce` could not be invoked directly. |
| `uv run kgtp data` | Exit 0; printed seven source configuration entries and produced no data. |
| `uv run kgtp graph` | Exit 0; printed `No processed graph found yet`. |
| `uv run kgtp baselines` | Exit 0; printed expected result filenames only. |
| `uv run kgtp heterodata` | Exit 1; missing `data/processed/nodes.parquet`. |
| `uv run kgtp splits` | Exit 1; missing `data/processed/heterodata/heterodata.pt`. |
| `uv run kgtp train --max-epochs 1` | Exit 1; missing `data/processed/splits/splits.pt`. |
| `uv run kgtp neo4j` | Exit 1; `Settings` rejected the top-level `project` field. |
| `uv run kgtp ablate` | Exit 1; all four required ablation JSON reports were missing. |
| `uv run kgtp explain` | Exit 0; printed that no HeteroData artifact exists. |
| `uv run python -m kgtp.cli smoke-train --tiny --output-dir <temp>` | Passed on the synthetic graph. It reported AUPRC `0.2398` and filtered MRR `0.4056`. These are smoke-fixture values, not biomedical benchmark results. |

## What is currently credible

The following implementation claims are supported by code and tests:

- deterministic table normalization helpers for several expected source shapes;
- Ensembl gene ID normalization and synthetic crosswalk tests;
- canonical node/edge table construction;
- PyG `HeteroData` construction from already processed tables;
- direct removal of held-out positive edges from message-passing edge indices;
- deterministic random negative exclusion across split partitions;
- working HGT, relation-wise GraphSAGE, homogeneous GraphSAGE, and custom
  relation-specific GNN forward passes on toy graphs;
- a functioning single-seed synthetic training loop;
- deterministic sampled binary and filtered tail-ranking metrics;
- an optional FastAPI application and Dockerfile that built in the latest CI;
- explicit biomedical and clinical disclaimers.

These are necessary components, but they do not yet establish a reproducible
biomedical target-prioritization benchmark.

## Blockers and unresolved uncertainties

- No production or sample source snapshots were available, so source schemas,
  release URLs, row counts, mapping yields, graph size, and runtime/memory cannot
  be validated.
- No checkpoint or full result artifact was available, so model quality,
  calibration, ranking stability, biological validity, and ablation conclusions
  remain unknown.
- Neo4j behavior beyond configuration parsing was not tested because processed
  tables are absent and the configuration currently fails validation.
- The local Docker daemon was unavailable; local image construction and runtime
  API behavior inside the image remain unverified. The corresponding remote CI
  image build is successful.
- The claim that this is the "first" open benchmark was not substantiated by the
  repository and was not treated as verified.
- DisGeNET redistribution policy was not independently analyzed in this phase;
  exclusion is conservative and does not block the proposed open sample
  pipeline.

## Recommended starting point

Phase 2 should begin with a small, redistributable, source-shaped dataset and a
single real orchestration path that works from a clean clone:

```text
fetch/copy sample inputs -> verify manifest -> normalize each source ->
build crosswalk -> enforce mapping gates -> assemble canonical graph ->
write processed tables and graph statistics
```

Do not start by modifying GNN architecture. Until a deterministic sample
pipeline produces auditable tables and manifests, later leakage, sampler,
baseline, training, and API fixes cannot be tested end to end.

## External sources checked

- Open Targets license: https://platform-docs.opentargets.org/licence
- STRING access and license: https://string-db.org/cgi/access
- Reactome license: https://reactome.org/license
- ChEMBL: https://www.ebi.ac.uk/chembl/
- UniProt license: https://www.uniprot.org/help/license
- Gene Ontology citation and license:
  https://geneontology.org/docs/go-citation-policy/
- Latest audited CI run:
  https://github.com/lysyloxidase/biomedical-kg-target-prioritization/actions/runs/27381598533
