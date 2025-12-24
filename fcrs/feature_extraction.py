from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from fcrs.preprocessing import FaceSample


@dataclass(frozen=True)
class FeatureExtractor:
    embedding_dim: int = 128
    model: Callable[[np.ndarray], np.ndarray] | None = None

    def extract(self, samples: Iterable[FaceSample]) -> dict[str, np.ndarray]:
        embeddings: dict[str, np.ndarray] = {}
        for sample in samples:
            if self.model is None:
                rng = np.random.default_rng(abs(hash(sample.sample_id)) % (2**32))
                embedding = rng.standard_normal(self.embedding_dim)
            else:
                embedding = self.model(sample.face_array)
            embeddings[sample.sample_id] = _l2_normalize(embedding)
        return embeddings


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
