# Model Card

## Model Family

The sample runner trains HGT, heterogeneous GraphSAGE, homogeneous GraphSAGE,
and R-GCN link predictors. Each model uses task-specific decoders for the
primary disease-gene task and optional drug-gene and gene-pathway tasks.

## Training Data

Models train on a message graph from which train supervision, validation, and
test target edges and their reverse edges are removed. Structural features are
fitted on that train graph. The setting remains transductive for node identities
and non-target relations.

## Intended Use

Checkpoints support software verification and computational hypothesis
generation on the sample graph. They are not validated biomedical models and
must not be used for clinical decisions.

## Selection And Evaluation

Best checkpoints are selected using primary-task validation full-candidate
AUPRC. Test results are computed only after selection. Five fixed seeds are
stored separately with dataset, split, feature, node-map, configuration, and
run hashes.

## Limitations

- Full-batch training does not scale to production graphs.
- Sample supervision is sparse and source-biased.
- Unlabeled pairs may contain unknown positives.
- Sigmoid scores are not clinically calibrated probabilities.
- No external, temporal, prospective, or wet-lab validation is available.

## Explainability

Integrated Gradients and edge occlusion describe model sensitivity. They are
not causal mechanisms and do not establish biological evidence.
