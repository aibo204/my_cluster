from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture

from fcrs.storage import ClusterMeta


@dataclass
class GMMClusterer:
    n_components: int = 5
    confidence_level: float = 0.99
    min_component_weight: float = 0.02
    noise_posterior_threshold: float = 0.2
    random_state: int | None = 42

    def fit_predict(
        self, features: np.ndarray
    ) -> tuple[list[int | None], set[int], set[int], dict[int, ClusterMeta]]:
        if features.size == 0:
            return [], set(), set(), {}
        gmm = self._gmm_modeling(features)
        membership = self._confidence_membership(features, gmm)
        component_graph = self._build_component_graph(membership)
        component_to_cluster = self._merge_components(component_graph)
        labels = self._assign_labels(features, gmm, membership, component_to_cluster)
        unknown, noise = self._identify_unknown_and_noise(features, gmm)
        cluster_meta = self._cluster_statistics(
            features, labels, gmm, component_to_cluster
        )
        return labels, unknown, noise, cluster_meta

    def _gmm_modeling(self, features: np.ndarray) -> GaussianMixture:
        n_components = min(self.n_components, features.shape[0])
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
        )
        model.fit(features)
        return model

    def _confidence_membership(
        self, features: np.ndarray, model: GaussianMixture
    ) -> np.ndarray:
        n_samples, dimension = features.shape
        threshold = chi2.ppf(self.confidence_level, df=dimension)
        membership = np.zeros((n_samples, model.n_components), dtype=bool)
        for k in range(model.n_components):
            if model.weights_[k] < self.min_component_weight:
                continue
            diff = features - model.means_[k]
            cov_inv = np.linalg.inv(model.covariances_[k])
            mahalanobis = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            membership[:, k] = mahalanobis <= threshold
        return membership

    def _build_component_graph(self, membership: np.ndarray) -> dict[int, set[int]]:
        graph: dict[int, set[int]] = {
            idx: set() for idx in range(membership.shape[1])
        }
        for i in range(membership.shape[0]):
            components = np.where(membership[i])[0]
            for a in components:
                for b in components:
                    if a != b:
                        graph[a].add(b)
                        graph[b].add(a)
        return graph

    def _merge_components(self, graph: dict[int, set[int]]) -> dict[int, int]:
        visited: set[int] = set()
        mapping: dict[int, int] = {}
        cluster_id = 0
        for node in graph:
            if node in visited:
                continue
            stack = [node]
            visited.add(node)
            while stack:
                current = stack.pop()
                mapping[current] = cluster_id
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            cluster_id += 1
        return mapping

    def _assign_labels(
        self,
        features: np.ndarray,
        model: GaussianMixture,
        membership: np.ndarray,
        component_to_cluster: dict[int, int],
    ) -> list[int | None]:
        labels: list[int | None] = []
        for idx, feature in enumerate(features):
            valid_components = np.where(membership[idx])[0]
            if len(valid_components) == 0:
                labels.append(None)
                continue
            best_component = self._best_component(feature, model, valid_components)
            labels.append(component_to_cluster[best_component])
        return labels

    def _best_component(
        self,
        feature: np.ndarray,
        model: GaussianMixture,
        valid_components: np.ndarray,
    ) -> int:
        distances = []
        for comp in valid_components:
            diff = feature - model.means_[comp]
            cov_inv = np.linalg.inv(model.covariances_[comp])
            distances.append(diff @ cov_inv @ diff)
        return int(valid_components[int(np.argmin(distances))])

    def _identify_unknown_and_noise(
        self, features: np.ndarray, model: GaussianMixture
    ) -> tuple[set[int], set[int]]:
        dimension = features.shape[1]
        threshold = chi2.ppf(self.confidence_level, df=dimension)
        unknown: set[int] = set()
        noise: set[int] = set()
        posterior = model.predict_proba(features)
        for idx, feature in enumerate(features):
            distances = []
            for comp in range(model.n_components):
                diff = feature - model.means_[comp]
                cov_inv = np.linalg.inv(model.covariances_[comp])
                distances.append(diff @ cov_inv @ diff)
            if all(distance > threshold for distance in distances):
                unknown.add(idx)
            if posterior[idx].max() < self.noise_posterior_threshold:
                noise.add(idx)
        return unknown, noise

    def _cluster_statistics(
        self,
        features: np.ndarray,
        labels: list[int | None],
        model: GaussianMixture,
        component_to_cluster: dict[int, int],
    ) -> dict[int, ClusterMeta]:
        cluster_meta: dict[int, ClusterMeta] = {}
        for comp in range(model.n_components):
            cluster_id = component_to_cluster[comp]
            if cluster_id not in cluster_meta:
                cluster_meta[cluster_id] = ClusterMeta(
                    cluster_id=cluster_id,
                    size=0,
                    mean=np.zeros(features.shape[1]),
                    covariance=np.zeros((features.shape[1], features.shape[1])),
                )
            cluster_meta[cluster_id].mean += model.means_[comp]
            cluster_meta[cluster_id].covariance += model.covariances_[comp]
            cluster_meta[cluster_id].size += 1
        for meta in cluster_meta.values():
            meta.mean /= meta.size
            meta.covariance /= meta.size
        return cluster_meta
