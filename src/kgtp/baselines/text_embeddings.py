"""Text-embedding baseline with cached deterministic embeddings."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from kgtp.baselines.common import cosine, hashed_vector
from kgtp.data.common import PathLike
from kgtp.eval.metrics import Triple


class TextEmbeddingBaseline:
    """Score disease-gene links by cosine similarity of text embeddings."""

    def __init__(
        self,
        *,
        cache_path: PathLike | None = None,
        dim: int = 64,
        model_name: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.dim = dim
        self.model_name = model_name
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(self, descriptions: Mapping[str, str]) -> TextEmbeddingBaseline:
        """Load cached embeddings or compute deterministic text embeddings."""

        cached = self._load_cache()
        for entity_id, description in descriptions.items():
            if entity_id in cached:
                self.embeddings[entity_id] = np.asarray(cached[entity_id], dtype=float)
            else:
                self.embeddings[entity_id] = self._encode(description or entity_id)
        self._write_cache()
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity between head and tail descriptions."""

        head, _, tail = triple
        return cosine(self._embedding(head), self._embedding(tail))

    def _embedding(self, entity_id: str) -> np.ndarray:
        if entity_id not in self.embeddings:
            self.embeddings[entity_id] = hashed_vector(entity_id, dim=self.dim)
        return self.embeddings[entity_id]

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        return cast(
            dict[str, list[float]],
            json.loads(self.cache_path.read_text(encoding="utf-8")),
        )

    def _encode(self, text: str) -> np.ndarray:
        if self.model_name is None:
            return hashed_vector(text, dim=self.dim)
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError:
            return hashed_vector(text, dim=self.dim)
        model = module.SentenceTransformer(self.model_name)
        return np.asarray(model.encode(text), dtype=float)

    def _write_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value.tolist() for key, value in sorted(self.embeddings.items())
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
