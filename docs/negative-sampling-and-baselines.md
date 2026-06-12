# Negative Sampling and Baselines

Phase 4 adds executable, train-only unlabeled sampling and honestly named
baseline models to the sample pipeline. These artifacts validate software
behavior on a small redistributable graph; they are not biomedical benchmark
results.

## Unlabeled Sampling

All samplers exclude every disease-gene positive in the full reference graph,
including validation and test positives. Their outputs are therefore described
as unlabeled pairs, not confirmed biological negatives.

- `random_unlabeled` uses bounded rejection/index sampling and never creates a
  table for the full Cartesian product.
- `degree_matched_unlabeled` matches positive targets using standardized
  train-only degree, PageRank, GO annotation count, and pathway count.
- `hard_unlabeled` draws only from a source-specific train-derived pool. A
  candidate must share a pathway or GO annotation with a training positive, or
  be its PPI neighbor. An undersized pool raises an error; there is no random
  fallback.

The generated pairs and diagnostics are stored under
`artifacts/sample/splits/negative_sampling/`. Diagnostics include the seed,
sample count, property distance, hard-pool size, fallback count, and label
semantics.

## Executed Baselines

`kgtp train-baselines` trains and evaluates:

- random score, target popularity, and source-score-only controls;
- logistic regression with random, degree-matched, and hard unlabeled pairs;
- gradient-boosted decision stumps and a feature-only MLP;
- matrix factorization and `AdjacencySVDBaseline`;
- Node2Vec with biased second-order walks, `p`, `q`, context windows, and
  skip-gram negative sampling;
- `HashTextBaseline`;
- native TransE, DistMult, ComplEx, and RotatE implementations.

Every arm uses the same held-out test positives, candidate genes, known-positive
filter, and ranking evaluator. Models and metrics are stored under
`artifacts/sample/models/baselines/` and
`artifacts/sample/metrics/baselines/`.

`SentenceTransformerBaseline` and `PubMedBERTBaseline` are separate optional
arms. They fail clearly when their dependency or requested weights are
unavailable and never substitute hash vectors. The sample run records their
availability in `models/baselines/availability.json`.

## Limitations

- The sample graph is too small for scientific model comparison.
- Sample hyperparameters are chosen for deterministic, fast execution rather
  than optimization.
- The known-positive registry is incomplete, so an unlabeled pair can still be
  an unknown true association.
- Native KGE implementations are formula-tested but are not a replacement for
  a production-scale, multi-seed benchmark with tuned losses and calibration.
