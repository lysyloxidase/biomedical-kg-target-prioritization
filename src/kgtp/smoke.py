"""Tiny deterministic graph and smoke-training utilities for CI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike
from kgtp.hetero.splits import leakage_free_random_link_split
from kgtp.models.train import TrainingConfig, train_one_seed


@dataclass(frozen=True)
class SmokeTrainResult:
    """Minimal smoke-train output for CI and CLI reporting."""

    metrics: dict[str, float]
    best_epoch: int
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "best_epoch": self.best_epoch,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }


def tiny_heterodata() -> HeteroData:
    """Build a small heterogeneous KG with all production node/edge families."""

    data = HeteroData()
    generator = torch.Generator().manual_seed(13)
    data["disease"].x = torch.eye(3)
    data["disease"].node_id = ["EFO_0004616", "EFO_0004617", "EFO_0004618"]
    data["disease"].label = [
        "knee osteoarthritis",
        "hip osteoarthritis",
        "hand osteoarthritis",
    ]
    data["gene"].x = torch.randn(8, 5, generator=generator)
    data["gene"].node_id = [
        "ENSG_GDF5",
        "ENSG_MMP13",
        "ENSG_FRZB",
        "ENSG_ADAMTS5",
        "ENSG_COL2A1",
        "ENSG_WNT5A",
        "ENSG_ACAN",
        "ENSG_NOVEL",
    ]
    data["gene"].symbol = [
        "GDF5",
        "MMP13",
        "FRZB",
        "ADAMTS5",
        "COL2A1",
        "WNT5A",
        "ACAN",
        "NOVEL",
    ]
    data["drug"].x = torch.randn(4, 4, generator=torch.Generator().manual_seed(17))
    data["drug"].node_id = [f"CHEMBL{i}" for i in range(4)]
    data["pathway"].x = torch.randn(5, 2, generator=torch.Generator().manual_seed(19))
    data["pathway"].node_id = [
        "R-HSA-WNT",
        "R-HSA-BMP",
        "R-HSA-ECM",
        "R-HSA-NFKB",
        "R-HSA-MAPK",
    ]
    data["pathway"].label = [
        "Wnt beta-catenin signaling",
        "TGF-beta BMP cartilage differentiation",
        "extracellular matrix collagen remodeling",
        "NF-kB inflammatory signaling",
        "MAPK signaling",
    ]
    data["go_term"].x = torch.randn(4, 4, generator=torch.Generator().manual_seed(23))
    data["go_term"].node_id = [f"GO:000000{i}" for i in range(4)]

    disease_gene = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 2, 3, 4, 4, 5, 6]],
        dtype=torch.long,
    )
    drug_gene = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    gene_pathway = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 0, 2, 2, 0, 1, 3]],
        dtype=torch.long,
    )
    gene_gene = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 6, 7, 2]], dtype=torch.long)
    gene_go = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 0, 1]], dtype=torch.long)

    data[("disease", "associated_with", "gene")].edge_index = disease_gene
    data[("gene", "rev_associated_with", "disease")].edge_index = disease_gene.flip(0)
    data[("drug", "targets", "gene")].edge_index = drug_gene
    data[("gene", "rev_targets", "drug")].edge_index = drug_gene.flip(0)
    data[("gene", "participates_in", "pathway")].edge_index = gene_pathway
    data[("pathway", "rev_participates_in", "gene")].edge_index = gene_pathway.flip(0)
    data[("gene", "interacts", "gene")].edge_index = gene_gene
    data[("gene", "annotated_with", "go_term")].edge_index = gene_go
    data[("go_term", "rev_annotated_with", "gene")].edge_index = gene_go.flip(0)
    return data


def run_smoke_train(
    *,
    seed: int = 13,
    max_epochs: int = 1,
    output_dir: PathLike | None = None,
) -> SmokeTrainResult:
    """Train HGT briefly on the tiny graph and return filtered metrics."""

    data = tiny_heterodata()
    bundle = leakage_free_random_link_split(
        data,
        seed=seed,
        num_val=0.2,
        num_test=0.2,
        disjoint_train_ratio=0.5,
        train_neg_sampling_ratio=1.0,
        eval_neg_sampling_ratio=1.0,
    )
    result = train_one_seed(
        bundle.train_data,
        bundle.val_data,
        bundle.test_data,
        data,
        seed=seed,
        config=TrainingConfig(
            model_name="hgt",
            hidden_channels=8,
            num_heads=2,
            num_layers=1,
            max_epochs=max_epochs,
            patience=max_epochs,
            negatives_per_positive=4,
        ),
        output_dir=output_dir,
    )
    return SmokeTrainResult(
        metrics=result.metrics,
        best_epoch=result.best_epoch,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
