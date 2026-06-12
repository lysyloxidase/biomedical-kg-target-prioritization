# Final Validation And Release-Readiness Report

Validation date: **2026-06-12**

Recommended release status: **alpha**

## Repository Status

- Branch: `main`
- HEAD: `4982c60e08ef8162f70cb0f614e901515bf81215`
- Remote tracking: `main` was aligned with `origin/main` before validation.
- Git tags: none.
- Historical repository commits visible locally: two.
- Package version: `0.2.0`
- FastAPI/OpenAPI version: `0.2.0`
- Validation host: Windows 11, Python 3.12.10, `uv` 0.10.12.
- The Phase 1-8 refactor is not committed. The worktree contains modified and
  untracked implementation, tests, sample data, configuration, and
  documentation.

This was a clean-environment and clean-artifact simulation, not a literal clone
of `origin/main`. A literal clone of the current remote would not contain the
uncommitted refactor. Commit and review the complete change set before calling
the repository reproducible from a fresh checkout.

## Clean-Checkout Simulation

The following required commands or their PowerShell equivalent were executed:

```text
git status
make clean-sample
Remove-Item -LiteralPath <verified-workspace>/.venv -Recurse -Force
uv sync --frozen --extra dev --no-editable
make reproduce-small
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright src tests
uv run pytest -q
uv run pytest --cov=kgtp --cov-report=term-missing
docker build -f docker/Dockerfile -t kgtp-final .
```

Before removal, `.venv` was resolved and checked to be exactly the expected
directory under the repository root. The locked environment was then recreated
from scratch.

Additional verification commands included:

```text
uv run pytest tests/integration/test_api_safety.py -q -vv
uv run pytest tests/unit/test_leakage_prevention.py tests/unit/test_splits.py tests/unit/test_unlabeled_sampling.py -q
uv run pytest tests/unit/test_baselines.py -q -vv
uv run pytest tests/unit/test_models.py -q -vv
uv run pytest tests/unit/test_explain.py -q -vv
uv run pytest tests/unit/test_eval_metrics.py -q -vv
uv run pytest tests/unit/test_quality_gates.py -q -vv
uv run bandit -q -r src -lll
uv run pip-audit --ignore-vuln CVE-2025-3000
uv run mkdocs build --strict
uv build
git diff --check
```

The final Docker image was also exercised with:

- validated artifacts mounted read-only;
- no artifacts;
- explicit `KGTP_DEMO_MODE=true`.

Incompatible graph, feature, run-manifest, untrained-model, missing-validation,
missing-checkpoint, and corrupted-checkpoint cases were exercised by the API
integration suite.

## Successful Checks

### Environment And Pipeline

- Fresh `uv sync`: 125 locked packages installed successfully.
- `make reproduce-small`: completed successfully.
- All nine stages reported `status=completed`.
- Stage output counts were:
  - prepare-sample: 1
  - normalize: 7
  - assemble: 9
  - split: 30
  - features: 4
  - train-baselines: 52
  - train-gnn: 106
  - evaluate: 21
  - report: 2
- Every stage output recorded in its manifest existed, was non-empty, and
  matched its declared byte count and SHA-256 hash.

The completed run manifest was:

```text
run_id: 20260612T182852Z-ae3710f1
status: completed
dirty_worktree: true
seeds: [13, 17, 19, 23, 29]
checkpoint hashes: 20
```

The current manifest and its immutable
`manifests/runs/<run_id>.json` archive were identical.

### Tests And Static Analysis

- Full test suite: **87 passed**, 3 dependency deprecation warnings.
- Coverage run: **87 passed**, total displayed coverage **84%**
  (`83.7%` before display rounding).
- Ruff lint: passed.
- Ruff format check: all 90 files formatted.
- mypy: no issues in 73 source files.
- pyright: 0 errors, 0 warnings.
- Leakage, split, determinism, and unlabeled-sampling selection: 24 passed.
- API safety integration suite: 10 passed.
- Baselines: 8 passed.
- GNN models and checkpoints: 9 passed.
- Explainability: 9 passed.
- Evaluation metrics: 8 passed.
- Documentation and security quality gates: 6 passed.
- `git diff --check`: passed.

Risk-based module coverage remained above configured thresholds:

| Module | Coverage |
| --- | ---: |
| split protocol | 75% |
| feature transformers | 85% |
| evaluation metrics | 93% |
| artifact validation | 83% |
| API safety | 89% |
| sample pipeline | 84% |

### Docker And API

- `kgtp-final` built successfully.
- Image user: `kgtp`.
- Image size: 2,967,860,424 bytes.
- Valid mounted artifacts: `/health` returned HTTP 200 with `trained=true`.
- Missing artifacts: `/health` returned HTTP 503 with a missing-checkpoint
  reason.
- Demo mode: `/predict` returned HTTP 200 and prominent `DEMO MODE` and
  non-scientific warnings.
- Incompatible artifact and checkpoint cases returned HTTP 503 in integration
  tests.

### Packaging, Documentation, And Security

- Wheel built:
  `dist/biomedical_kg_target_prioritization-0.2.0-py3-none-any.whl`.
- Source distribution built:
  `dist/biomedical_kg_target_prioritization-0.2.0.tar.gz`.
- MkDocs strict build passed.
- Bandit passed at the configured high-severity threshold.
- `pip-audit` reported no other known vulnerabilities after the single,
  documented `CVE-2025-3000` exception.
- Package and API versions both report `0.2.0`.

## Acceptance Checklist

| Requirement | Result | Evidence or qualification |
| --- | --- | --- |
| `make reproduce-small` succeeds from a clean state | Qualified pass | Succeeds after removing artifacts and `.venv`; source worktree is not committed, so a literal remote checkout is not yet equivalent. |
| Sample dataset is legal and documented | Qualified pass | Seven files match recorded hashes; provenance and CC0/CC-BY/CC-BY-SA terms are documented. This is a source-license review, not legal advice. |
| Every pipeline stage performs real work | Pass | Nine completed stage manifests with 232 verified output records in total. |
| Split happens before feature fitting | Pass | Stage order and feature manifest declare `fit_scope=train_message_graph_only`. |
| Validation and test edges do not affect train features | Pass | Feature and PageRank invariance tests passed. |
| Leakage tests pass | Pass | Held-out, reverse-edge, train-supervision, corrupted-split, duplicate-edge, and auxiliary-edge tests passed. |
| Random, degree-matched, and hard sampling are distinct | Pass | Distribution and strategy-rule tests passed. |
| Known positives are never sampled as unlabeled | Pass | Full-reference and held-out-positive protection tests passed. |
| Baseline names match implementations | Pass | Adjacency SVD and Node2Vec are separate classes and tested as distinct algorithms. |
| Node2Vec is real or correctly renamed | Pass | Node2Vec uses biased second-order walks, `p`/`q`, context windows, and skip-gram negative sampling. |
| Transformer models never silently hash-fallback | Pass | Optional models fail explicitly; hash text is a separately named baseline. |
| KGE scoring and data separation are correct | Pass | Hand-computed TransE, DistMult, ComplEx, and RotatE scoring tests passed; fitting uses train examples, validation selects state, and test positives are evaluation-only. |
| GNN checkpoints represent trained models | Pass | 20 non-empty checkpoints and 20 configs with `trained=true`; untrained evaluation is rejected. |
| Models share one evaluation protocol | Pass | Four GNN families share one split hash; 16 baseline arms share six test positives and filtered evaluation. |
| Explanations require trained compatible artifacts | Pass | Untrained, mismatched, and held-out-edge tests passed. |
| API fails closed | Pass | Missing, corrupted, untrained, incomplete, and incompatible cases return 503. |
| Artifacts contain hashes and manifests | Pass | Run, stage, dataset, graph, split, feature, node-map, checkpoint, and dependency hashes were verified. |
| Tests, lint, and type checks pass | Pass | All required local commands returned code 0. |
| Docker builds | Pass | `kgtp-final` built and served both validated and fail-closed modes. |
| README matches capabilities | Pass | Quality gates passed and README explicitly limits claims to the sample pipeline. |
| No sample numbers are presented as scientific findings | Pass | Benchmark and README explicitly label sample metrics as software-validation output. |

## Failed Checks And Warnings

No user-required validation command failed.

Non-product exploratory audit probes initially made incorrect assumptions about
JSON structure:

- `data/sample/dataset_manifest.json` was queried, but the documented source
  manifest is `data/sample/manifest.json`.
- Two temporary scripts treated manifest mappings as lists or path strings.
- Two temporary scripts expected nested fields at the top level or expected
  `identical_split_hash` to be Boolean rather than the common hash value.

Corrected probes passed. These were audit-script errors and did not identify a
repository defect.

Observed warnings:

- PyTorch reports `torch.jit.script` deprecation during import.
- FastAPI's test client reports a Starlette/httpx compatibility deprecation.
- Material for MkDocs emits an upstream MkDocs 2.0 migration warning.

## Optional Dependencies Not Tested End To End

- `SentenceTransformerBaseline`: unavailable because
  `sentence-transformers` is not installed.
- `PubMedBERTBaseline`: unavailable because `transformers` and model weights
  are not installed.
- Their no-fallback and mocked real-encoder paths are unit tested, but actual
  external model downloads and inference were not tested.
- No CUDA/GPU training run was executed.
- No live Neo4j service or credentialed deployment was tested; Neo4j remains
  optional.
- No production external-source refresh was executed during final validation.

## Unresolved Issues

1. **The refactor is uncommitted.** A real fresh clone of `origin/main` cannot
   reproduce these results until the complete change set is committed.
2. **No remote CI result exists for this change set.** Local command equivalents
   pass, but GitHub Actions has not evaluated the uncommitted files.
3. **No historical release baseline exists.** The repository has no tags,
   signed releases, migration history, or published artifact checksums.
4. **The production/full-data pipeline remains incomplete.** `make reproduce`
   intentionally fails rather than making a false reproducibility claim.
5. **The Docker image is large.** The locked Linux PyTorch resolution includes
   CUDA-related packages even though the sample run is CPU-capable.
6. **One security exception remains.** `CVE-2025-3000` is narrowly ignored and
   documented; it must be revisited when advisory ranges or a fixed dependency
   become available.
7. **Legal review is not external counsel.** ChEMBL share-alike and all upstream
   attribution obligations must remain attached to redistributed sample data.

## Full-Data Limitations

- External source APIs and licenses can change.
- Several source versions are live snapshots rather than immutable archives.
- Full-data acquisition, storage sizing, retries, and checksum manifests have
  not been exercised end to end.
- The sample uses one disease and 30 disease-gene positives.
- Full-batch GNN training is unsuitable for a production-scale graph.
- Production-scale neighbor sampling, distributed training, and GPU memory
  requirements are not established.
- Full-data hyperparameter selection and temporal validation are absent.
- Neo4j credentials are required only for the optional Neo4j path and must be
  supplied through environment variables.

## Scientific Limitations

- Successful execution does not establish biological validity.
- Unobserved pairs are unlabeled and may contain true associations.
- The graph and literature sources contain popularity and curation biases.
- The protocol is transductive for node identities and non-target relations.
- Five seeds describe implementation variability but do not support strong
  significance claims.
- Explanation scores are model attributions, not causal mechanisms.
- No prospective, experimental, clinical, or independent external validation
  was performed.
- Sample benchmark numbers must not be used to rank biomedical methods or
  justify target-development decisions.

## Release Recommendation

**Alpha**

The repository is suitable for an explicitly pre-1.0 alpha release focused on
software evaluation of the small redistributable sample. The executable
pipeline, split-first safeguards, sampling tests, trained checkpoints,
artifact validation, fail-closed API, packaging, documentation, and Docker
image all passed local verification.

It is not a beta, release candidate, or 1.0-ready scientific package because
the refactor is uncommitted, remote CI has not run on it, no historical release
or tag exists, full-data execution is incomplete, optional transformer arms
were not run with real weights, GPU/full-scale behavior is unknown, and no
biological validation has been performed.

Before publishing the alpha, commit the complete refactor, run the pinned CI
workflow on that commit, create a signed/tagged release with checksums, and
retain the existing scientific and licensing disclaimers.
