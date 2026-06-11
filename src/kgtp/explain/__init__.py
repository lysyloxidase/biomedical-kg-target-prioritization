"""Phase 6 explanation utilities."""

from kgtp.explain.attention import AttentionWeight, extract_hgt_attention_weights
from kgtp.explain.case_studies import (
    CaseStudyResult,
    PredictionCandidate,
    build_known_target_case_study,
    build_novel_prediction_case_study,
    build_phase6_case_studies,
)
from kgtp.explain.explainer import DISEASE_GENE_EDGE, TargetExplainer
from kgtp.explain.metapaths import MetaPathExplanation, PathNode, rank_metapaths

__all__ = [
    "DISEASE_GENE_EDGE",
    "AttentionWeight",
    "CaseStudyResult",
    "MetaPathExplanation",
    "PathNode",
    "PredictionCandidate",
    "TargetExplainer",
    "build_known_target_case_study",
    "build_novel_prediction_case_study",
    "build_phase6_case_studies",
    "extract_hgt_attention_weights",
    "rank_metapaths",
]
