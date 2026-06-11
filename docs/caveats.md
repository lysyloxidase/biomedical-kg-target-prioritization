# Caveats

1. Open Targets scores aggregate heterogeneous evidence and should not be treated
   as direct effect sizes.
2. Knee OA evidence is incomplete; all-OA genetics can leak non-knee biology.
3. Source release dates differ, so edges are not temporally synchronized.
4. STRING confidence is not equivalent to physical interaction certainty.
5. PPI expansion can over-represent highly studied genes.
6. GO annotation depth varies by gene and evidence code.
7. Reactome coverage favors curated pathway biology.
8. ChEMBL MoA edges can mix direct binding, functional modulation, and target
   family annotations.
9. UniProt and Ensembl mappings are versioned and can drift.
10. Ensembl-gene hub normalization may collapse biologically relevant isoform
    detail.
11. Hatzikotoulas 2025 is all-OA, so knee-only claims should prefer
    `EFO_0004616` evidence.
12. The graph is intentionally focused and should not be interpreted as a global
    biomedical KG.
13. Link-prediction performance may reflect graph topology artifacts rather than
    therapeutic plausibility.
14. Target prioritization outputs are computational hypotheses requiring
    independent experimental and clinical validation.
