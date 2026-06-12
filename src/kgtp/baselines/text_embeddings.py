"""Explicit text-only baselines without silent model substitution."""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from kgtp.baselines.common import cosine, hashed_vector
from kgtp.data.common import PathLike
from kgtp.eval.metrics import Triple


class HashTextBaseline:
    """Deterministic signed feature hashing over entity descriptions."""

    def __init__(self, *, dim: int = 64) -> None:
        self.dim = dim
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(self, descriptions: Mapping[str, str]) -> HashTextBaseline:
        """Hash every supplied description into an explicit text feature vector."""

        self.embeddings = {
            entity_id: hashed_vector(description or entity_id, dim=self.dim)
            for entity_id, description in sorted(descriptions.items())
        }
        return self

    def score(self, triple: Triple) -> float:
        """Return description-vector cosine similarity."""

        head, _, tail = triple
        if head not in self.embeddings or tail not in self.embeddings:
            return 0.0
        return cosine(self.embeddings[head], self.embeddings[tail])


class SentenceTransformerBaseline:
    """Sentence-transformer text baseline requiring its real dependency/model."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_path: PathLike | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(
        self,
        descriptions: Mapping[str, str],
    ) -> SentenceTransformerBaseline:
        """Encode descriptions or fail clearly when the model is unavailable."""

        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            msg = (
                "SentenceTransformerBaseline requires the optional "
                "'sentence-transformers' dependency"
            )
            raise RuntimeError(msg) from exc
        try:
            model = module.SentenceTransformer(self.model_name)
            ids = sorted(descriptions)
            vectors = model.encode([descriptions[entity_id] for entity_id in ids])
        except Exception as exc:
            msg = f"Unable to load or run sentence-transformer model {self.model_name}"
            raise RuntimeError(msg) from exc
        self.embeddings = {
            entity_id: np.asarray(vector, dtype=float)
            for entity_id, vector in zip(ids, vectors, strict=True)
        }
        self._write_cache()
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity, with no hash fallback."""

        head, _, tail = triple
        if head not in self.embeddings or tail not in self.embeddings:
            return 0.0
        return cosine(self.embeddings[head], self.embeddings[tail])

    def _write_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {key: value.tolist() for key, value in sorted(self.embeddings.items())},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class PubMedBERTBaseline:
    """Mean-pooled PubMedBERT baseline requiring Hugging Face transformers."""

    def __init__(
        self,
        *,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    ) -> None:
        self.model_name = model_name
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(self, descriptions: Mapping[str, str]) -> PubMedBERTBaseline:
        """Load tokenizer/model and mean-pool real contextual embeddings."""

        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            msg = (
                "PubMedBERTBaseline requires the optional 'transformers' "
                "dependency and model weights"
            )
            raise RuntimeError(msg) from exc
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModel.from_pretrained(self.model_name)
            model.eval()
            ids = sorted(descriptions)
            for entity_id in ids:
                encoded = tokenizer(
                    descriptions[entity_id],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                with torch.no_grad():
                    output = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                self.embeddings[entity_id] = np.asarray(
                    pooled.squeeze(0).cpu().numpy(), dtype=float
                )
        except Exception as exc:
            msg = f"Unable to load or run PubMedBERT model {self.model_name}"
            raise RuntimeError(msg) from exc
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity, with no fallback implementation."""

        head, _, tail = triple
        if head not in self.embeddings or tail not in self.embeddings:
            return 0.0
        return cosine(self.embeddings[head], self.embeddings[tail])


def optional_text_model_status() -> dict[str, dict[str, Any]]:
    """Report optional text arms without importing model weights."""

    return {
        "sentence_transformer": {
            "available": importlib.util.find_spec("sentence_transformers") is not None,
            "dependency": "sentence-transformers",
        },
        "pubmedbert": {
            "available": importlib.util.find_spec("transformers") is not None,
            "dependency": "transformers",
        },
    }


def load_embedding_cache(path: PathLike) -> dict[str, np.ndarray]:
    """Load an explicit embedding cache for tests and external tooling."""

    payload = cast(
        dict[str, list[float]],
        json.loads(Path(path).read_text(encoding="utf-8")),
    )
    return {key: np.asarray(value, dtype=float) for key, value in payload.items()}
