from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FeatureRecord:
    sample_id: str
    feature: np.ndarray
    cluster_id: int | None = None
    status: str = "normal"


@dataclass
class ClusterMeta:
    cluster_id: int
    size: int
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class FeatureStore:
    records: dict[str, FeatureRecord] = field(default_factory=dict)
    clusters: dict[int, ClusterMeta] = field(default_factory=dict)

    def add_record(self, record: FeatureRecord) -> None:
        self.records[record.sample_id] = record

    def update_clusters(self, clusters: dict[int, ClusterMeta]) -> None:
        self.clusters = clusters

    def save(self, folder: str | Path) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        records_path = folder_path / "records.json"
        arrays_path = folder_path / "features.npy"
        sample_ids = list(self.records.keys())
        features = np.stack([self.records[sid].feature for sid in sample_ids])
        np.save(arrays_path, features)
        records_payload = {
            "sample_ids": sample_ids,
            "records": [
                {
                    "sample_id": record.sample_id,
                    "cluster_id": record.cluster_id,
                    "status": record.status,
                }
                for record in self.records.values()
            ],
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "mean": cluster.mean.tolist(),
                    "covariance": cluster.covariance.tolist(),
                }
                for cluster in self.clusters.values()
            ],
        }
        records_path.write_text(json.dumps(records_payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, folder: str | Path) -> "FeatureStore":
        folder_path = Path(folder)
        records_path = folder_path / "records.json"
        arrays_path = folder_path / "features.npy"
        payload = json.loads(records_path.read_text())
        features = np.load(arrays_path)
        sample_ids = payload["sample_ids"]
        store = cls()
        for idx, record_payload in enumerate(payload["records"]):
            store.records[record_payload["sample_id"]] = FeatureRecord(
                sample_id=record_payload["sample_id"],
                feature=features[idx],
                cluster_id=record_payload["cluster_id"],
                status=record_payload["status"],
            )
        clusters: dict[int, ClusterMeta] = {}
        for cluster_payload in payload["clusters"]:
            cluster_id = cluster_payload["cluster_id"]
            clusters[cluster_id] = ClusterMeta(
                cluster_id=cluster_id,
                size=cluster_payload["size"],
                mean=np.array(cluster_payload["mean"]),
                covariance=np.array(cluster_payload["covariance"]),
            )
        store.clusters = clusters
        return store
