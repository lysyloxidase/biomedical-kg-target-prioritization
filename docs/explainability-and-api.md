# Explainability and Safe API

Phase 6 permits prediction and explanation only after a trained artifact bundle
passes compatibility validation. The normal runtime has no untrained or
smoke-graph fallback.

## Required artifacts

The loader requires:

- a trained checkpoint and saved model configuration;
- the train-message `HeteroData` graph;
- dataset, graph, feature, and split manifests;
- a node-index map and fitted feature-transformer state;
- persisted validation metrics.
- a completed run manifest binding the artifacts to one execution.

It verifies file hashes, semantic graph and split hashes, model configuration,
node-index assignments, feature fit scope, run/checkpoint compatibility, and
the absence of train, validation, and test supervision positives from the
message graph. Any failure stops loading.

Run the checked sample explanation workflow after `make reproduce-small`:

```bash
make explain
```

The command writes full-candidate rankings, model attributions, and evidence
cards under `artifacts/sample/report/explanations/`.

## Interpretation

Integrated Gradients describes feature sensitivity relative to a zero-feature
baseline. Edge occlusion reports prediction-score changes after removing local
message edges. Both are model attributions, not causal explanations or
biological validation.

Endpoint-conditioned topology summaries are labeled as proxies and are not
reported as learned HGT attention. Candidate cards include available pathways,
PPI neighbors, GO terms, drug information, source provenance, uncertainty
limitations, and a computational-hypothesis warning.

## API behavior

`make api` uses the validated sample artifacts by default. `/health`,
`/graph-data`, `/graph-stats`, and `/predict` return HTTP 503 when the artifact
bundle is missing, corrupt, untrained, smoke-only, or incompatible.

Prediction responses include checkpoint and dataset hashes, model and split
identifiers, validation metrics, candidate protocol, trained status, and
warnings.

For UI and transport demonstrations only:

```bash
KGTP_DEMO_MODE=true make api
```

Demo mode trains a one-epoch model on a synthetic smoke graph in memory. Its
responses are prominently marked non-scientific and must not be used for
biomedical interpretation.

## Residual limitations

- PyTorch checkpoints are pickle-based and must come from a trusted source.
- Explanations use one checkpoint and do not estimate predictive uncertainty.
- The small sample is incomplete and cannot support efficacy claims.
- Attribution stability across architectures, cohorts, and biological
  perturbations is not established.
