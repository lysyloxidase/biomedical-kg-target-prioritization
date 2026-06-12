"""Feature-only MLP baseline without graph message passing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from kgtp.baselines.common import pair_feature_vector
from kgtp.eval.metrics import Triple


class FeatureMLPBaseline:
    """Train a pair-feature MLP using no graph encoder."""

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 100,
        seed: int = 13,
    ) -> None:
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.node_features: Mapping[str, Sequence[float]] | None = None
        self.model: nn.Sequential | None = None

    def fit(
        self,
        positives: Sequence[Triple],
        negatives: Sequence[Triple],
        *,
        node_features: Mapping[str, Sequence[float]],
        validation_positives: Sequence[Triple] = (),
        validation_negatives: Sequence[Triple] = (),
    ) -> FeatureMLPBaseline:
        """Fit on train pairs and select parameters using validation loss."""

        self.node_features = node_features
        train_x, train_y = _pair_batch(
            positives, negatives, node_features=node_features
        )
        validation = (
            _pair_batch(
                validation_positives,
                validation_negatives,
                node_features=node_features,
            )
            if validation_positives or validation_negatives
            else None
        )
        torch.manual_seed(self.seed)
        self.model = nn.Sequential(
            nn.Linear(train_x.size(1), self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(train_x).squeeze(-1)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if validation is None:
                    selection_loss = float(loss.item())
                else:
                    validation_logits = self.model(validation[0]).squeeze(-1)
                    selection_loss = float(
                        criterion(validation_logits, validation[1]).item()
                    )
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, triple: Triple) -> float:
        """Return feature-only MLP logit."""

        if self.model is None or self.node_features is None:
            msg = "Feature MLP must be fitted before scoring"
            raise RuntimeError(msg)
        vector = pair_feature_vector(triple, self.node_features)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(tensor).item())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return learned parameters for persistence."""

        if self.model is None:
            msg = "Feature MLP has no fitted state"
            raise RuntimeError(msg)
        return self.model.state_dict()


def _pair_batch(
    positives: Sequence[Triple],
    negatives: Sequence[Triple],
    *,
    node_features: Mapping[str, Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    triples = [*positives, *negatives]
    matrix = np.vstack(
        [pair_feature_vector(triple, node_features) for triple in triples]
    )
    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    return (
        torch.tensor(matrix, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
