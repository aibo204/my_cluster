from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    min_face_size: int = 20
    confidence_threshold: float = 0.9


@dataclass(frozen=True)
class FaceSample:
    sample_id: str
    image_path: Path
    face_array: np.ndarray


def _mock_detect_and_align(image_path: Path) -> np.ndarray | None:
    if not image_path.exists():
        return None
    rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
    return rng.random((112, 112, 3), dtype=np.float32)


def preprocess_faces(
    image_paths: Iterable[str | Path], config: PreprocessConfig
) -> list[FaceSample]:
    samples: list[FaceSample] = []
    for idx, path in enumerate(image_paths):
        image_path = Path(path)
        face = _mock_detect_and_align(image_path)
        if face is None:
            continue
        if min(face.shape[:2]) < config.min_face_size:
            continue
        sample_id = f"sample_{idx:06d}"
        samples.append(
            FaceSample(sample_id=sample_id, image_path=image_path, face_array=face)
        )
    return samples
