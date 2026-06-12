# biomedical-kg-target-prioritization

Research software for OA-centric biomedical knowledge-graph target
prioritization. Version `0.2.0` provides an executable, redistributable sample
pipeline. It does not provide a completed full-scale benchmark or validated
drug-target discoveries.

Computational hypothesis generation only. This repository does not diagnose
disease, recommend treatment, establish causality, or provide clinical advice.

## Current Status

Implemented and tested:

- a checked-in knee-OA sample dataset with source and license metadata;
- identifier normalization and canonical Parquet graph assembly;
- split-first disease-gene, drug-gene, and gene-pathway supervision;
- train-message-only structural features and PyG `HeteroData`;
- sampled-unlabeled strategies, baseline models, and four GNN families;
- five-seed evaluation, trained-artifact explanations, and a fail-closed API;
- run, dataset, graph, split, feature, checkpoint, and stage manifests.

Not completed:

- production-scale source acquisition and graph construction;
- full-data hyperparameter selection and external validation;
- prospective or experimental biological validation;
- scientific claims based on the sample metrics.

No real full-scale benchmark results are included in this checkout.

## Task Definition

The primary link-prediction task is:

```text
disease -> associated_with -> gene
```

The sample query is knee osteoarthritis, `EFO_0004616`. Eligible genes are
ranked with full-candidate evaluation. Other known disease-gene positives are
filtered for ranking metrics. Drug-gene and gene-pathway links are optional
auxiliary tasks with explicit loss weights.

Pairs absent from the registry are called `unlabeled`, not biological
negatives. The repository contains no curated negative set.

## Quickstart

Requirements: Python 3.11 or newer, `uv`, and a platform supported by PyTorch.
Neo4j is optional.

```bash
git clone https://github.com/lysyloxidase/biomedical-kg-target-prioritization
cd biomedical-kg-target-prioritization
make setup
make reproduce-small
make test
```

`make reproduce-small` performs real local work and writes the current manifest
to `artifacts/sample/manifests/run.json` plus an immutable run-id-specific copy
under `artifacts/sample/manifests/runs/`. The manifest records the Git state,
locked dependencies, effective configuration, source versions and licenses,
input hashes, graph/split/feature/checkpoint hashes, seeds, command, and status.

`make reproduce` intentionally fails because the production workflow is not
yet complete.

## Sample Dataset

`data/sample/` is a deterministic, incomplete OA-oriented snapshot containing
232 nodes and six biomedical edge tables. It combines attributed records from
Open Targets, STRING, Reactome, ChEMBL, UniProt, Ensembl, and GOA/QuickGO.

The sample is intended for software integration, leakage tests, and tutorial
execution. It is not a comprehensive OA knowledge base. See
[`docs/dataset-card.md`](docs/dataset-card.md) and
[`docs/dataset-card-sample.md`](docs/dataset-card-sample.md).

## Pipeline Architecture

```text
checked-in sample
  -> normalize identifiers
  -> assemble full reference graph
  -> split target and auxiliary relations
  -> construct train message graph
  -> fit feature transformer on train graph
  -> build shared PyG views
  -> train baselines and GNNs
  -> evaluate held-out supervision
  -> report, explain, and serve validated artifacts
```

The full reference graph is reserved for provenance, known-positive checks, and
filtered evaluation. It is not used for training-time structural features.

## Leakage Protection

Validation and test target edges, including reverse edges, are absent from the
train message graph. Degree, PageRank, GO vocabulary, annotation statistics,
and hard sampled-unlabeled pools are fitted from train-permitted information.
Split and feature hashes are checked before training, evaluation, explanation,
and API serving.

This is a transductive protocol with shared node identities and non-target
relations. It is not described as universally leakage-free. Exact guarantees
and residual risks are in
[`docs/leakage-prevention.md`](docs/leakage-prevention.md).

## Implemented Models

Sample pipeline:

- random, degree/popularity, source-score, logistic regression, gradient
  boosting, matrix factorization, and feature MLP;
- `AdjacencySVDBaseline`;
- true Node2Vec using biased walks and skip-gram training;
- hash-text baseline;
- DistMult and ComplEx;
- HGT, relation-wise heterogeneous GraphSAGE, homogeneous GraphSAGE control,
  and R-GCN.

The code also provides explicit TransE and RotatE scoring implementations.
Sentence Transformer and PubMedBERT are optional arms. They fail clearly and
are marked unavailable when their dependencies or weights are absent; they
never silently fall back to hash vectors. In the default locked environment,
both optional transformer arms are unavailable.

## Evaluation

All models use the same split and candidate protocol. Reports distinguish:

- full-candidate ranking over all eligible genes;
- random, degree-matched, and hard sampled-unlabeled evaluation.

Metrics include AUPRC, AUROC, MRR, Hits@K, Precision@K, Recall@K, NDCG@K,
enrichment, lift over popularity, and calibration where probabilities exist.
Five fixed seeds produce raw metrics, means, standard deviations, confidence
intervals, paired bootstrap intervals, and sign-flip comparisons.

AUPRC values from different candidate prevalence must not be compared without
qualification. See
[`docs/evaluation-protocol.md`](docs/evaluation-protocol.md).

## Artifacts

The sample run writes:

```text
artifacts/sample/
  manifests/       run, source, dataset, and stage manifests
  normalized/      normalized Parquet tables
  graph/           full reference graph
  train_message_graph/
  splits/          supervision, registry, and sampled-unlabeled sets
  features/        PyG graph, node maps, fitted transformer
  models/          baseline and per-seed GNN checkpoints
  metrics/         per-model, per-seed, and comparison metrics
  report/          benchmark and explanation outputs
```

Generated artifacts are not committed. Sample metrics validate executable
plumbing and must not be presented as scientific benchmark results.

## Explainability And API

```bash
make explain
make api
```

Integrated Gradients and edge occlusion are model attributions, not causal
explanations. Evidence cards include pathways, PPI neighbors, GO terms, drug
information, provenance, uncertainty limitations, and hypothesis warnings.

The API returns HTTP 503 unless checkpoint, graph, features, split, validation,
node map, dataset, and run manifests are present and mutually compatible.
Synthetic demo behavior requires `KGTP_DEMO_MODE=true` and is prominently
marked non-scientific. See [`docs/api-safety.md`](docs/api-safety.md).

## Full-Data Prerequisites

A future full run requires:

- legally permitted access to every upstream source and acceptance of its
  current terms;
- pinned releases, raw checksums, redistribution review, and sufficient disk;
- production neighbor sampling or graph partitioning;
- a documented compute budget and model-selection protocol;
- temporal or otherwise justified evaluation splits;
- independent biological and statistical review.

Neo4j credentials are optional and must be supplied through environment
variables. See `.env.example`; never commit `.env`.

## Licenses

Code is Apache-2.0. Source data retain upstream terms. The sample includes CC0,
CC-BY-4.0, and CC-BY-SA-3.0 inputs; ChEMBL-derived redistribution therefore
requires attention to share-alike obligations. DisGeNET is excluded from the
redistributable sample.

See [`docs/data-sources.md`](docs/data-sources.md) and the dataset cards.

## Limitations

The sample is small, literature and curation bias remain, unlabeled pairs may
be unknown positives, and non-target context is transductive and not temporally
partitioned. Checkpoints use trusted-local PyTorch serialization. Explanation
stability and probability calibration are not established for clinical use.

See [`docs/biomedical-limitations.md`](docs/biomedical-limitations.md).
