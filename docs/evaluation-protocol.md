# Evaluation Protocol

## Primary Task

For each disease query, rank eligible genes for
`disease -> associated_with -> gene`.

## Candidate Protocols

Full-candidate evaluation scores every eligible gene. Other registered
positives for the query are filtered when computing ranking metrics.

Sampled-unlabeled evaluation uses separately persisted random, degree-matched,
and train-only hard sets. These pairs are not treated as confirmed negatives.
Candidate prevalence is stored with every result.

## Metrics

Reports include AUPRC, AUROC, MRR, Hits@1/3/10/50, Precision@K, Recall@K,
NDCG@K, enrichment factor, lift over popularity, Brier score, and expected
calibration error when a probability-like output exists.

AUPRC values from protocols with different positive prevalence are not directly
comparable.

## Model Comparison

The sample runner uses seeds `13, 17, 19, 23, 29`, identical splits, and
per-seed checkpoints. It reports means, sample standard deviations, approximate
confidence intervals, paired bootstrap intervals, paired t-tests with stated
assumptions, and exact sign-flip comparisons.

Five seeds do not justify strong significance claims. Sample metrics are
execution evidence, not biomedical benchmark conclusions.

Implementation details are also recorded in
[`gnn-evaluation-protocol.md`](gnn-evaluation-protocol.md).
