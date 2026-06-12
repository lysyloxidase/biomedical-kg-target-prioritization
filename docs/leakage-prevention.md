# Leakage Prevention

## Scope

The sample workflow protects training-time graph structure and fitted features
from validation and test edges of the primary target relation:

```text
Disease -[associated_with]-> Gene
```

The protection is implemented by splitting canonical Parquet edges before
constructing PyG objects or fitting topology-derived features.

## Enforced Data Flow

```text
normalized canonical graph
    -> target-edge assignments
    -> train message graph
    -> feature-transformer fit
    -> shared train-fitted node features
    -> train/validation/test PyG views
    -> training and held-out evaluation
```

All three PyG views use exactly the same train message edges, reverse edges, node
index maps, and train-fitted features. They differ only in their supervision
tables.

The full reference graph is not passed to the encoder or the feature
transformer. Evaluation receives a restricted object containing:

- known positive triples for filtered ranking;
- deterministic node IDs defining the candidate universe.

## Protected Signals

Validation and test disease-gene edges cannot affect:

- total degree used in gene features;
- PageRank used in gene features;
- message-passing neighborhoods;
- connected topology visible to the GNN encoder;
- train message edges or reverse message edges;
- GO vocabulary selection;
- pathway participant counts and pathway depth;
- GO annotation counts and information content;
- feature dimensions or feature-transformer state;
- random training-unlabeled generation, with the known-positive registry used
  to avoid assigning an already known edge label `0`;
- model initialization or training supervision.

No dimensionality reduction, feature scaling, or feature normalization is
currently fitted. If introduced later, it must be part of the same
`fit(train_graph)` transformer state and covered by perturbation tests.

## Full Reference Graph Policy

The full graph is retained as `artifacts/sample/graph/` for provenance. Its
disease-gene registry is copied to
`splits/full_known_positives.parquet`.

During model execution it may be used only to:

- filter other known positives from ranking candidates;
- reject sampled unlabeled pairs that are already known positives;
- identify whether an output candidate is already known;
- verify provenance and graph hashes.

It is not an encoder input and is not consulted by
`TrainGraphFeatureTransformer`.

## Tests

`tests/unit/test_leakage_prevention.py` verifies:

- deleting a test edge does not change any training feature tensor;
- deleting a validation edge does not change gene features;
- deleting a test edge does not change training PageRank;
- reverse test edges are absent from all message graphs;
- train supervision edges are absent from the train message graph;
- transformer state records the train message graph hash, not the full graph
  hash;
- identical seeds reproduce assignments and sampled unlabeled pairs;
- different seeds change assignments;
- duplicate reference edges and deliberately corrupted assignments are
  rejected.

The integration test additionally executes the complete sample pipeline and
checks the separate graph, split, transformer, model, metric, run-manifest, and
report artifacts. Training and evaluation validate hashes in upstream stage
and run manifests before using them.

## Remaining Risks

The protections above do not eliminate every form of scientific leakage or
bias:

- The setting is transductive. All node identities and all non-target relations
  are visible in every split.
- Non-target relations are not temporally partitioned. A pathway, GO, PPI, or
  drug-target edge published after a disease-gene association could encode
  future knowledge.
- Source curation and literature popularity may correlate with held-out labels.
- The known-positive registry is incomplete. A sampled unlabeled pair can be a
  real but unrecorded association.
- Disease-gene is the primary task. Drug-gene and gene-pathway are auxiliary
  tasks with separate split policies and tests.
- Hyperparameter search is not implemented. Future search must use validation
  supervision only and must never inspect test metrics.
- Phase 4 hard-negative pools are fitted from train-message PPI, pathway, and
  GO relations around training positives. Validation and test labels are not
  used to construct the pools.
- Phase 5 auxiliary drug-gene and gene-pathway supervision is partitioned before
  feature fitting; held-out canonical and reverse edges are absent from the
  shared train message graph.

For these reasons, documentation describes the concrete guarantees rather than
claiming universal freedom from leakage.
