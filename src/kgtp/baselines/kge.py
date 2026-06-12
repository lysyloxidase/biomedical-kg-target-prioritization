"""Mathematically distinct native TransE, DistMult, ComplEx, and RotatE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import nn

from kgtp.eval.metrics import Triple

KGEName = Literal["TransE", "DistMult", "ComplEx", "RotatE"]
KGE_MODELS: tuple[KGEName, ...] = ("TransE", "DistMult", "ComplEx", "RotatE")


def score_kge_vectors(
    model_name: KGEName,
    head_real: torch.Tensor,
    relation_real: torch.Tensor,
    tail_real: torch.Tensor,
    *,
    head_imag: torch.Tensor | None = None,
    relation_imag: torch.Tensor | None = None,
    tail_imag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score indexed vectors using the named KGE equation."""

    if model_name == "TransE":
        return -(head_real + relation_real - tail_real).abs().sum(dim=-1)
    if model_name == "DistMult":
        return (head_real * relation_real * tail_real).sum(dim=-1)
    if head_imag is None or tail_imag is None:
        msg = f"{model_name} requires imaginary entity components"
        raise ValueError(msg)
    if model_name == "ComplEx":
        if relation_imag is None:
            msg = "ComplEx requires imaginary relation components"
            raise ValueError(msg)
        return (
            head_real * relation_real * tail_real
            + head_imag * relation_real * tail_imag
            + head_real * relation_imag * tail_imag
            - head_imag * relation_imag * tail_real
        ).sum(dim=-1)
    rotated_real = head_real * torch.cos(relation_real) - head_imag * torch.sin(
        relation_real
    )
    rotated_imag = head_real * torch.sin(relation_real) + head_imag * torch.cos(
        relation_real
    )
    return -torch.sqrt(
        (rotated_real - tail_real).square()
        + (rotated_imag - tail_imag).square()
        + 1e-12
    ).sum(dim=-1)


class KGEBaseline:
    """Train a named KGE model with train/validation separation."""

    def __init__(
        self,
        *,
        model_name: KGEName = "TransE",
        dimension: int = 16,
        learning_rate: float = 0.02,
        epochs: int = 100,
        seed: int = 13,
    ) -> None:
        if model_name not in KGE_MODELS:
            msg = f"Unsupported KGE model: {model_name}"
            raise ValueError(msg)
        self.model_name: KGEName = model_name
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.entity_index: dict[str, int] = {}
        self.relation_index: dict[str, int] = {}
        self.model: _KGEParameters | None = None
        self.best_validation_loss: float | None = None

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
        *,
        validation_positives: Sequence[Triple] = (),
        validation_negatives: Sequence[Triple] = (),
    ) -> KGEBaseline:
        """Fit using train examples and select state only on validation examples."""

        train_examples = [*positives, *negatives]
        validation_examples = [*validation_positives, *validation_negatives]
        all_index_examples = [*train_examples, *validation_examples]
        entities = sorted(
            {entity for head, _, tail in all_index_examples for entity in (head, tail)}
        )
        relations = sorted({relation for _, relation, _ in all_index_examples})
        if not entities or not relations:
            msg = "KGE training requires entities and relations"
            raise ValueError(msg)
        self.entity_index = {entity: index for index, entity in enumerate(entities)}
        self.relation_index = {
            relation: index for index, relation in enumerate(relations)
        }
        torch.manual_seed(self.seed)
        self.model = _KGEParameters(
            self.model_name,
            num_entities=len(entities),
            num_relations=len(relations),
            dimension=self.dimension,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        train_triples, train_labels = self._batch(positives, negatives)
        validation_batch = (
            self._batch(validation_positives, validation_negatives)
            if validation_examples
            else None
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model.score_indices(train_triples)
            loss = criterion(logits, train_labels)
            loss.backward()
            optimizer.step()
            self.model.project_constraints()

            if validation_batch is None:
                validation_loss = float(loss.item())
            else:
                with torch.no_grad():
                    validation_logits = self.model.score_indices(validation_batch[0])
                    validation_loss = float(
                        criterion(validation_logits, validation_batch[1]).item()
                    )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_validation_loss = best_loss
        return self

    def score(self, triple: Triple) -> float:
        """Return the named model's compatibility score."""

        if self.model is None:
            msg = "KGE model must be fitted before scoring"
            raise RuntimeError(msg)
        head, relation, tail = triple
        if (
            head not in self.entity_index
            or tail not in self.entity_index
            or relation not in self.relation_index
        ):
            return 0.0
        indices = torch.tensor(
            [
                [
                    self.entity_index[head],
                    self.relation_index[relation],
                    self.entity_index[tail],
                ]
            ],
            dtype=torch.long,
        )
        with torch.no_grad():
            return float(self.model.score_indices(indices).item())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return learned tensors for checkpoint persistence."""

        if self.model is None:
            msg = "KGE model has no fitted state"
            raise RuntimeError(msg)
        return self.model.state_dict()

    def metadata(self) -> dict[str, object]:
        """Return algorithm and validation-selection metadata."""

        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "seed": self.seed,
            "entity_index": self.entity_index,
            "relation_index": self.relation_index,
            "best_validation_loss": self.best_validation_loss,
        }

    def _batch(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        triples = [*positives, *negatives]
        indices = torch.tensor(
            [
                [
                    self.entity_index[head],
                    self.relation_index[relation],
                    self.entity_index[tail],
                ]
                for head, relation, tail in triples
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [1.0] * len(positives) + [0.0] * len(negatives),
            dtype=torch.float32,
        )
        return indices, labels


class _KGEParameters(nn.Module):
    def __init__(
        self,
        model_name: KGEName,
        *,
        num_entities: int,
        num_relations: int,
        dimension: int,
    ) -> None:
        super().__init__()
        self.model_name: KGEName = model_name
        self.dimension = dimension
        self.entity_real = nn.Embedding(num_entities, dimension)
        self.relation_real = nn.Embedding(num_relations, dimension)
        if model_name in {"ComplEx", "RotatE"}:
            self.entity_imag: nn.Embedding | None = nn.Embedding(
                num_entities, dimension
            )
        else:
            self.entity_imag = None
        if model_name == "ComplEx":
            self.relation_imag: nn.Embedding | None = nn.Embedding(
                num_relations, dimension
            )
        else:
            self.relation_imag = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6.0 / np.sqrt(max(1, self.dimension))
        nn.init.uniform_(self.entity_real.weight, -bound, bound)
        nn.init.uniform_(self.relation_real.weight, -bound, bound)
        if self.entity_imag is not None:
            nn.init.uniform_(self.entity_imag.weight, -bound, bound)
        if self.relation_imag is not None:
            nn.init.uniform_(self.relation_imag.weight, -bound, bound)

    def score_indices(self, triples: torch.Tensor) -> torch.Tensor:
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        head_real = self.entity_real(heads)
        relation_real = self.relation_real(relations)
        tail_real = self.entity_real(tails)
        head_imag = self.entity_imag(heads) if self.entity_imag is not None else None
        relation_imag = (
            self.relation_imag(relations) if self.relation_imag is not None else None
        )
        tail_imag = self.entity_imag(tails) if self.entity_imag is not None else None
        return score_kge_vectors(
            self.model_name,
            head_real,
            relation_real,
            tail_real,
            head_imag=head_imag,
            relation_imag=relation_imag,
            tail_imag=tail_imag,
        )

    def project_constraints(self) -> None:
        with torch.no_grad():
            if self.model_name == "RotatE":
                self.relation_real.weight.copy_(
                    torch.remainder(
                        self.relation_real.weight + torch.pi,
                        2 * torch.pi,
                    )
                    - torch.pi
                )
            entity_norm = self.entity_real.weight.norm(dim=1, keepdim=True)
            self.entity_real.weight.div_(entity_norm.clamp(min=1.0))
            if self.entity_imag is not None:
                imag_norm = self.entity_imag.weight.norm(dim=1, keepdim=True)
                self.entity_imag.weight.div_(imag_norm.clamp(min=1.0))
