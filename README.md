# biomedical-kg-target-prioritization

The first open, reproducible heterogeneous-GNN link-prediction benchmark scaffold
for knee osteoarthritis target prioritization: HGT, GraphSAGE, R-GCN, seven
non-graph baselines, leakage-free splits, filtered OGB-style metrics, empirical
ablations, and interpretable target rationales.

Author: `lysyloxidase`

Computational hypothesis generation for target prioritization only. This is not
validated drug-target discovery, not clinical advice, and not a treatment
recommendation.

## Quickstart

```bash
git clone https://github.com/lysyloxidase/biomedical-kg-target-prioritization
cd biomedical-kg-target-prioritization
docker compose -f docker/docker-compose.yml up -d
make setup
make reproduce
```

For CI-scale verification without the full OA data build:

```bash
make test
make smoke-train
make api
```

`make api` serves the optional FastAPI app at `http://localhost:8000` with:

- `GET /health`
- `GET /graph-stats`
- `GET /predict?disease=EFO_0004616&top_k=20`

The API loads `data/processed/heterodata/heterodata.pt` and
`reports/models/hgt_seed13/model.pt` when present. Otherwise it falls back to a
tiny deterministic smoke graph so the service can be tested on a fresh clone.

## Headline Results

Primary metric: AUPRC. AUROC is reported but optimistic under sparse-link
imbalance. Ranking metrics are filtered in the OGB/KG style. The current checkout
does not include saved full-run `reports/results_*.json` artifacts, so the table
below is intentionally marked pending rather than filled with synthetic numbers.

Disease->gene, mean +/- std over at least five seeds, filtered:

| Model | AUROC | AUPRC | Hits@10 | MRR |
|---|---:|---:|---:|---:|
| Popularity (degree) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| Logistic Regression (features) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| Matrix Factorization | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| Node2Vec | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| Text embeddings (PubMedBERT) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| TransE / DistMult / ComplEx / RotatE | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| GraphSAGE (to_hetero) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| R-GCN | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| **HGT (hero)** | pending full OA run | pending full OA run | pending full OA run | pending full OA run |

## Ablations

Every cell should be mean +/- std over at least five seeds and regenerated from
saved per-seed reports under `reports/ablations/`.

**Ablation 1+2: no-KG vs KG vs KG+text**

| Setting | AUROC | AUPRC | Hits@10 | MRR |
|---|---:|---:|---:|---:|
| no-KG (LR/MLP, features) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| KG (HGT, structural) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| KG+text (HGT + PubMedBERT) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |

**Ablation 3: homogeneous vs relational vs heterogeneous**

| Setting | AUROC | AUPRC | Hits@10 | MRR |
|---|---:|---:|---:|---:|
| GraphSAGE (homogeneous / to_hetero) | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| R-GCN | pending full OA run | pending full OA run | pending full OA run | pending full OA run |
| HGT | pending full OA run | pending full OA run | pending full OA run | pending full OA run |

**Ablation 4: design knobs**

The design grid crosses negative sampling (`random`, `degree_matched`, `hard`),
layers (`1`, `2`, `3`), and node features (`one-hot`, `structural`, `ESM`,
`text`). The table is emitted as markdown and LaTeX by `make ablate` from saved
Phase 5 reports. Random negatives are expected to inflate AUROC relative to hard
negatives; AUPRC and filtered MRR are the more credible signals.

## Graph Statistics

The production graph-statistics table is emitted by `make graph` and by
`GET /graph-stats`. Pending full OA graph build:

| Node type | Count |   | Edge type | Count |
|---|---:|---|---|---:|
| Disease | pending full OA build |   | associated_with | pending full OA build |
| Gene | pending full OA build |   | interacts (PPI) | pending full OA build |
| Pathway | pending full OA build |   | participates_in | pending full OA build |
| Drug | pending full OA build |   | targets | pending full OA build |
| GOTerm | pending full OA build |   | annotated_with | pending full OA build |

Additional graph statistics: density, mean degree, degree distribution, and
positive disease->gene link count.

## Interpretability Case Study

Phase 6 generates known-target and novel-prediction rationales:

- Known-target sanity checks: `GDF5` and `MMP13` when present in the graph.
- Expected biology: Wnt/beta-catenin, TGF-beta/BMP, cartilage and extracellular
  matrix pathways, and OA-relevant PPI neighbors.
- Novel predictions are explicitly flagged as computational hypotheses, not
  validated targets.
- Figures are saved to `reports/figures/` as PNG/PDF explanatory subgraphs.

Run:

```bash
make explain
```

## Honest Findings

This project is designed to report what the ablations actually show:

- If HGT does not beat logistic regression on node features, the headline is
  that graph message passing adds little under this sparse supervision setting.
- If text embeddings dominate, the narrative should pivot to text co-occurrence
  being the main signal and quantify what the graph adds.
- If R-GCN or GraphSAGE match HGT, the result is evidence that heterogeneous
  attention is not necessary for this OA graph.

Absolute MRR/Hits should not be compared directly with ogbl-biokg leaderboard
numbers because this OA graph is intentionally tiny. The contribution is the
controlled comparison, leakage prevention, and interpretability layer rather
than state-of-the-art leaderboard performance.

## Data Sources And Licenses

| Source | Role | License / policy |
|---|---|---|
| Open Targets | disease-target associations | CC0 |
| STRING v12 | PPI expansion | CC-BY-4.0 |
| Reactome | pathways and hierarchy | CC-BY-4.0 |
| ChEMBL 35 | drug-target mechanisms | CC-BY-SA-3.0 |
| UniProt | gene/protein annotation | CC-BY-4.0 |
| GO / GOA | GO terms and annotations | CC-BY-4.0 |

DisGeNET is excluded because the 2024 freemium redistribution policy is not
compatible with this open benchmark. Ground-truth context is documented in
`docs/oa-biology.md`; data caveats are documented in `docs/caveats.md`.

## Reproducibility

- Pinned source releases and fixed seeds.
- Saved node-index maps and split seeds.
- Leakage-free `RandomLinkSplit` with disjoint message-passing and supervision
  edges.
- Split-leakage tests run explicitly in CI.
- Smoke-train gate runs HGT on a tiny synthetic heterogeneous graph.
- Docker build gate validates the production image.
- `make reproduce` runs the full pipeline targets end to end when data artifacts
  and saved reports are available.

## Canonical Edge Types

| Source type | Relation | Target type | Source |
|---|---|---|---|
| Disease | associated_with | Gene | Open Targets |
| Gene | interacts | Gene | STRING |
| Gene | participates_in | Pathway | Reactome |
| Drug | targets | Gene | ChEMBL |
| Gene | annotated_with | GOTerm | GOA |
| Pathway | parent_of | Pathway | Reactome optional |

## License

Code is Apache-2.0. Figures are CC-BY-4.0. Source data retain their upstream
licenses; see `docs/data-sources.md`.
