# Split Protocol

## Prediction Task

The Phase 3 sample task predicts canonical
`Disease -[associated_with]-> Gene` edges. Other relations are fixed
transductive context:

- `Gene -[interacts]-> Gene`
- `Gene -[participates_in]-> Pathway`
- `Drug -[targets]-> Gene`
- `Gene -[annotated_with]-> GOTerm`
- `Pathway -[parent_of]-> Pathway`

This policy is recorded in `split_metadata.json`.

## Canonicalization Preconditions

Splitting occurs after identifier normalization and graph assembly. The target
table must:

- use the canonical Disease/associated_with/Gene schema;
- contain both endpoints in the node table;
- contain no duplicate `(source_id, target_id)` pair;
- contain enough edges for non-empty message, train, validation, and test
  positive partitions.

Violations raise an exception and produce a nonzero CLI exit.

## Positive Partitions

With seed 13, the 30 sample disease-gene edges are deterministically shuffled
and partitioned using:

- test fraction: 0.20;
- validation fraction: 0.20 after test removal;
- train-supervision fraction: 0.30 of the remaining train pool;
- remaining train edges: message-passing edges.

For the checked-in sample this yields:

| Partition | Positive edges |
| --- | ---: |
| Train message graph | 13 |
| Train supervision | 6 |
| Validation supervision | 5 |
| Test supervision | 6 |

The four positive partitions are pairwise disjoint and reconstruct the complete
known-positive registry.

## Negative Supervision

One random unlabeled pair is generated per positive for each supervision split.
Sampling:

- uses deterministic split-specific seeds;
- draws from the fixed disease and gene node universe;
- excludes every edge in the full known-positive registry;
- excludes unlabeled pairs already assigned to another split;
- does not use degree, PageRank, validation/test topology, or model scores.

This is a software-validation protocol. Sampled pairs are unlabeled and are not
claimed to be biologically verified negatives.

Phase 5 applies the same split-before-features rule to the optional drug-gene
and gene-pathway tasks. Their canonical and reverse supervision edges are
removed from every train message graph before feature fitting.

## Message Graphs

`train_message_graph/` contains:

- all nodes;
- all non-target canonical edges;
- only the target edges assigned to the message partition.

Train, validation, and test PyG views all copy this same graph. Their reverse
`Gene -[rev_associated_with]-> Disease` edges are generated only from the 13
message edges. Supervision edges and their reverses are absent.

## Artifacts

```text
artifacts/sample/
├── graph/                         # full reference graph
├── train_message_graph/           # encoder and feature-fit graph
├── splits/
│   ├── full_known_positives.parquet
│   ├── split_assignments.parquet
│   ├── message_edges.parquet
│   ├── leakage_validation.json
│   ├── supervision/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── test.parquet
│   ├── split_metadata.json
│   └── splits.pt
└── features/
    ├── feature_transformer.json
    ├── heterodata.pt
    └── node_index_maps.json
```

`split_metadata.json` records:

- seed and split ratios;
- target-relation and non-target-relation policy;
- full reference graph hash;
- train message graph hash;
- deterministic node-index-map hash;
- per-partition semantic hashes and counts.

The hashes cover canonical dataframe content rather than Parquet container
bytes.

## Validation

Before feature fitting, the validator checks:

- registry/reference equality;
- valid partition names;
- no duplicate or multiply assigned canonical edge;
- pairwise partition disjointness;
- exact reconstruction of all reference positives;
- message-table consistency;
- supervision labels and positive assignments;
- unlabeled-pair exclusion and cross-split disjointness;
- graph, node-map, and partition hashes.

After PyG construction it also checks:

- identical message edges in all three views;
- identical feature tensors in all three views;
- exact reverse-edge correspondence;
- no positive supervision edge or reverse edge in message passing.
