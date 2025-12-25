from __future__ import annotations

from pathlib import Path

import numpy as np

from fcrs.clustering import GMMClusterer
from fcrs.feature_extraction import FeatureExtractor
from fcrs.preprocessing import PreprocessConfig, preprocess_faces
from fcrs.reporting import format_cluster_report
from fcrs.storage import FeatureRecord, FeatureStore


def main() -> None:
    image_paths = [Path("demo_images") / f"image_{idx}.jpg" for idx in range(20)]
    preprocess_config = PreprocessConfig(min_face_size=20, confidence_threshold=0.9)
    samples = preprocess_faces(image_paths, preprocess_config)
    extractor = FeatureExtractor(embedding_dim=128)
    features = extractor.extract(samples)

    feature_ids = list(features.keys())
    feature_matrix = (
        np.stack([features[sample_id] for sample_id in feature_ids])
        if feature_ids
        else np.empty((0, 128))
    )
    clusterer = GMMClusterer(n_components=3, confidence_level=0.99)
    labels, unknown, noise, cluster_meta = (
        clusterer.fit_predict(feature_matrix) if feature_matrix.size else ([], set(), set(), {})
    )

    store = FeatureStore()
    unknown_ids = {feature_ids[i] for i in unknown}
    for label, sample_id in zip(labels, feature_ids):
        feature = features[sample_id]
        status = "unknown" if sample_id in unknown_ids else "normal"
        record = FeatureRecord(
            sample_id=sample_id, feature=feature, cluster_id=label, status=status
        )
        store.add_record(record)
    store.update_clusters(cluster_meta)
    store.save("fcrs_output")

    print(format_cluster_report(labels, unknown, noise, cluster_meta))


if __name__ == "__main__":
    main()
