from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fcrs.clustering import GMMClusterer


@dataclass
class IncrementalClusterer:
    clusterer: GMMClusterer
    buffer_threshold: int = 50
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[int | None] = field(default_factory=list)
    unknown_buffer: list[np.ndarray] = field(default_factory=list)

    def add_samples(self, new_features: list[np.ndarray]) -> None:
        for feature in new_features:
            self._assign_or_buffer(feature)
        if len(self.unknown_buffer) >= self.buffer_threshold:
            self._recluster()

    def _assign_or_buffer(self, feature: np.ndarray) -> None:
        if not self.features:
            self.unknown_buffer.append(feature)
            return
        all_features = np.stack(self.features)
        labels, unknown, _, _ = self.clusterer.fit_predict(all_features)
        self.labels = labels
        model_features = np.stack(self.features)
        if not model_features.size:
            self.unknown_buffer.append(feature)
            return
        combined = np.vstack([model_features, feature])
        updated_labels, unknown, _, _ = self.clusterer.fit_predict(combined)
        feature_index = combined.shape[0] - 1
        if feature_index in unknown:
            self.unknown_buffer.append(feature)
        else:
            self.features.append(feature)
            self.labels = updated_labels[:-1]

    def _recluster(self) -> None:
        combined = self.features + self.unknown_buffer
        if not combined:
            return
        features = np.stack(combined)
        labels, _, _, _ = self.clusterer.fit_predict(features)
        self.features = list(features)
        self.labels = labels
        self.unknown_buffer = []
