from __future__ import annotations

from collections import Counter

from fcrs.storage import ClusterMeta


def format_cluster_report(
    labels: list[int | None],
    unknown: set[int],
    noise: set[int],
    cluster_meta: dict[int, ClusterMeta],
) -> str:
    counts = Counter(label for label in labels if label is not None)
    lines = ["聚类统计报告"]
    for cluster_id, count in counts.items():
        meta = cluster_meta.get(cluster_id)
        if meta:
            lines.append(
                f"- Cluster {cluster_id}: size={count}, "
                f"mean_norm={float(meta.mean.sum()):.4f}"
            )
        else:
            lines.append(f"- Cluster {cluster_id}: size={count}")
    lines.append(f"- Unknown samples: {len(unknown)}")
    lines.append(f"- Noise samples: {len(noise)}")
    return "\n".join(lines)
