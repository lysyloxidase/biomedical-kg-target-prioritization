# GNN and Evaluation Protocol

Phase 5 trains HGT, relation-wise heterogeneous GraphSAGE, a homogeneous
GraphSAGE control, and R-GCN on one shared split and feature artifact. The
sample run uses five fixed seeds and is intended to validate software behavior,
not establish biomedical model superiority.

## Tasks and Loss

The primary task is:

`disease -> associated_with -> gene`

Two optional auxiliary tasks are split before feature fitting:

- `drug -> targets -> gene`
- `gene -> participates_in -> pathway`

Their held-out canonical and reverse edges are absent from every message graph.
The default sample loss is:

```text
total_loss =
    1.00 * disease_gene_loss
  + 0.25 * drug_gene_loss
  + 0.25 * gene_pathway_loss
```

Each task has a separate decoder head and separate validation/test metrics.
Checkpoint selection uses only primary-task full-candidate validation AUPRC.

## Models

- HGT uses typed attention.
- Heterogeneous GraphSAGE uses a separate `SAGEConv` for every edge type and
  sums relation messages.
- Homogeneous GraphSAGE collapses node and edge types as a control.
- R-GCN uses relation-specific message transformations.

All encoders use configurable dropout, residual updates, and LayerNorm.
Training records deterministic seeds, device/runtime metadata, optimizer state,
best epoch, hyperparameters, dataset hash, split hash, message-graph hash,
feature-transformer hash, and node-index-map hash.

The sample uses full-batch training. This is suitable for the small graph only;
production graphs require neighbor sampling, graph partitioning, or another
bounded-memory loader.

## Evaluation Protocols

Primary full-candidate evaluation ranks every eligible gene. Other registered
positives for the query are filtered. Reports include:

- AUROC and average precision/AUPRC;
- MRR and Hits@1, Hits@3, Hits@10, Hits@50;
- Precision@K, Recall@K, NDCG@K, and enrichment factor;
- AUPRC lift and ratio over train-only target popularity;
- Brier score and expected calibration error for sigmoid GNN probabilities.

The same models are also evaluated against the persisted random,
degree-matched, and hard sampled-unlabeled sets. Every report stores candidate
prevalence and warns that AUPRC values from different prevalence/protocols are
not directly comparable.

Labels use the terms `positive` and `unlabeled`. The repository has no curated
biological negative set.

## Multi-Seed Comparison

The runner stores raw results for seeds `13, 17, 19, 23, 29`, plus means,
sample standard deviations, and normal-approximation 95% confidence intervals.
Paired model comparisons use identical seeds and include:

- a paired t-test with its normal-difference assumption stated;
- a deterministic paired bootstrap interval for the mean difference;
- an exact paired sign-flip permutation p-value.

Five seeds and this small graph do not support strong significance claims.

## PU Limitation

A class-prior-aware positive-unlabeled risk estimator is not implemented.
Training uses weighted binary cross-entropy where `0` denotes a sampled
unlabeled pair. This can bias probability calibration and should not be
interpreted as supervised learning from confirmed negatives. A defensible PU
extension requires a documented class-prior estimate and validation protocol
on a larger dataset.
