# Biomedical Limitations

This software prioritizes computational hypotheses; it does not validate
targets.

Key limitations:

1. The sample graph is deliberately small and incomplete.
2. Literature and database curation favor well-studied genes and pathways.
3. Missing links are unlabeled and may be true associations.
4. Disease definitions and source evidence can be heterogeneous.
5. Identifier mapping can discard or merge biologically distinct records.
6. Non-target graph context is transductive and not temporally partitioned.
7. Drug-target links do not imply efficacy, tissue exposure, safety, or
   disease modification.
8. Pathway, GO, PPI, and text evidence can reflect circular literature support.
9. Model probabilities are not calibrated for clinical decision-making.
10. Model attributions are not causal explanations.
11. Five random seeds are insufficient for strong statistical claims.
12. No external cohort, prospective experiment, replication study, or wet-lab
    validation is included.

Any candidate requires independent genetic, molecular, pharmacological, safety,
and clinical review.
