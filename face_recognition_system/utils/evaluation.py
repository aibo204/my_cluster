"""
评估工具
Evaluation Utilities for DGFC Face Recognition System

提供聚类和识别性能评估的各种指标
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, some metrics will be unavailable")


def calculate_clustering_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算聚类评估指标
    
    Args:
        labels_true: 真实标签
        labels_pred: 预测标签
        embeddings: 特征向量 (用于计算内部指标)
        
    Returns:
        指标字典
    """
    if not SKLEARN_AVAILABLE:
        logger.error("Scikit-learn is required for clustering metrics")
        return {}
    
    metrics = {}
    
    # 外部指标 (需要真实标签)
    if labels_true is not None and len(np.unique(labels_true)) > 1:
        metrics["ari"] = adjusted_rand_score(labels_true, labels_pred)
        metrics["nmi"] = normalized_mutual_info_score(labels_true, labels_pred)
        
        # Purity
        metrics["purity"] = calculate_purity(labels_true, labels_pred)
        
        # BCubed metrics
        bcubed = calculate_bcubed_metrics(labels_true, labels_pred)
        metrics.update(bcubed)
    
    # 内部指标 (不需要真实标签)
    if embeddings is not None and len(np.unique(labels_pred)) > 1:
        try:
            # 过滤掉未分配的点
            mask = labels_pred > 0
            if np.sum(mask) > 1 and len(np.unique(labels_pred[mask])) > 1:
                metrics["silhouette"] = silhouette_score(embeddings[mask], labels_pred[mask])
                metrics["calinski_harabasz"] = calinski_harabasz_score(embeddings[mask], labels_pred[mask])
                metrics["davies_bouldin"] = davies_bouldin_score(embeddings[mask], labels_pred[mask])
        except Exception as e:
            logger.warning(f"Failed to calculate internal metrics: {e}")
    
    # 聚类统计
    unique_labels = np.unique(labels_pred[labels_pred > 0])
    metrics["n_clusters"] = len(unique_labels)
    
    if len(unique_labels) > 0:
        cluster_sizes = [np.sum(labels_pred == l) for l in unique_labels]
        metrics["avg_cluster_size"] = np.mean(cluster_sizes)
        metrics["min_cluster_size"] = np.min(cluster_sizes)
        metrics["max_cluster_size"] = np.max(cluster_sizes)
    
    return metrics


def calculate_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    计算聚类纯度 (Purity)
    
    Purity = (1/N) * Σ max_j |cluster_i ∩ class_j|
    """
    contingency = defaultdict(lambda: defaultdict(int))
    
    for true_label, pred_label in zip(labels_true, labels_pred):
        contingency[pred_label][true_label] += 1
    
    total_correct = 0
    for cluster_counts in contingency.values():
        total_correct += max(cluster_counts.values())
    
    return total_correct / len(labels_true)


def calculate_bcubed_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Dict[str, float]:
    """
    计算BCubed精确率、召回率和F1
    
    BCubed是专门用于聚类评估的指标
    """
    n = len(labels_true)
    
    precision_sum = 0.0
    recall_sum = 0.0
    
    for i in range(n):
        same_cluster = labels_pred == labels_pred[i]
        same_class = labels_true == labels_true[i]
        
        # 同簇且同类的点数
        correct = np.sum(same_cluster & same_class)
        
        # BCubed Precision: 同簇中同类的比例
        cluster_size = np.sum(same_cluster)
        if cluster_size > 0:
            precision_sum += correct / cluster_size
        
        # BCubed Recall: 同类中同簇的比例
        class_size = np.sum(same_class)
        if class_size > 0:
            recall_sum += correct / class_size
    
    precision = precision_sum / n
    recall = recall_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": f1,
    }


def calculate_recognition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算识别评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_scores: 预测置信度分数 (用于计算ROC/AUC)
        
    Returns:
        指标字典
    """
    if not SKLEARN_AVAILABLE:
        return {}
    
    metrics = {}
    
    # 基础指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # 多分类指标
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        average = 'binary'
    else:
        average = 'weighted'
    
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Rank-based metrics
    if y_scores is not None:
        metrics.update(calculate_rank_metrics(y_true, y_scores))
    
    return metrics


def calculate_rank_metrics(
    y_true: np.ndarray,
    scores_matrix: np.ndarray,
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    计算Rank-based指标 (Rank-1, Rank-5, etc.)
    
    Args:
        y_true: 真实标签 (N,)
        scores_matrix: 分数矩阵 (N, num_classes)
        ks: K值列表
        
    Returns:
        指标字典
    """
    metrics = {}
    
    n_samples = len(y_true)
    n_classes = scores_matrix.shape[1]
    
    # 对每个样本，获取排序后的预测
    sorted_indices = np.argsort(-scores_matrix, axis=1)
    
    for k in ks:
        if k > n_classes:
            continue
        
        correct = 0
        for i in range(n_samples):
            if y_true[i] in sorted_indices[i, :k]:
                correct += 1
        
        metrics[f"rank_{k}_accuracy"] = correct / n_samples
    
    return metrics


def calculate_verification_metrics(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    计算人脸验证指标 (1:1比对)
    
    Args:
        pairs: 特征对列表
        labels: 是否同一人的标签 (1=same, 0=different)
        thresholds: 阈值列表
        
    Returns:
        指标字典，包括最佳阈值、准确率、FAR、FRR等
    """
    # 计算相似度
    similarities = np.array([
        np.dot(p[0], p[1]) / (np.linalg.norm(p[0]) * np.linalg.norm(p[1]) + 1e-10)
        for p in pairs
    ])
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    labels = np.array(labels)
    
    best_accuracy = 0
    best_threshold = 0.5
    far_at_best = 0
    frr_at_best = 0
    
    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        
        # True Positives, False Positives, etc.
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        accuracy = (tp + tn) / len(labels)
        
        # FAR (False Accept Rate) = FP / (FP + TN)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # FRR (False Reject Rate) = FN / (FN + TP)
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            far_at_best = far
            frr_at_best = frr
    
    return {
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "far": far_at_best,
        "frr": frr_at_best,
        "eer": (far_at_best + frr_at_best) / 2,  # 近似EER
    }


def evaluate_clustering(
    clustering_result,
    labels_true: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    综合评估聚类结果
    
    Args:
        clustering_result: ClusteringResult对象
        labels_true: 真实标签 (可选)
        embeddings: 特征向量 (可选)
        print_report: 是否打印报告
        
    Returns:
        评估结果字典
    """
    labels_pred = clustering_result.labels
    
    results = {}
    
    # 基础统计
    results["n_samples"] = len(labels_pred)
    results["n_clusters"] = clustering_result.n_clusters
    results["n_core_points"] = int(np.sum(clustering_result.core_mask))
    results["n_unassigned"] = int(np.sum(labels_pred == 0))
    
    # 聚类指标
    if labels_true is not None:
        metrics = calculate_clustering_metrics(labels_true, labels_pred, embeddings)
        results["metrics"] = metrics
    else:
        # 只计算内部指标
        if embeddings is not None:
            metrics = calculate_clustering_metrics(None, labels_pred, embeddings)
            results["metrics"] = metrics
    
    # 打印报告
    if print_report:
        print("\n" + "=" * 50)
        print("DGFC Clustering Evaluation Report")
        print("=" * 50)
        print(f"Total samples: {results['n_samples']}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Core points: {results['n_core_points']} ({100*results['n_core_points']/results['n_samples']:.1f}%)")
        print(f"Unassigned: {results['n_unassigned']} ({100*results['n_unassigned']/results['n_samples']:.1f}%)")
        
        if "metrics" in results:
            print("\nMetrics:")
            for name, value in results["metrics"].items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        
        print("=" * 50 + "\n")
    
    return results


def calculate_metrics(y_true, y_pred, embeddings=None, task="clustering"):
    """
    通用指标计算接口
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        embeddings: 特征向量
        task: 任务类型 ("clustering" or "recognition")
    """
    if task == "clustering":
        return calculate_clustering_metrics(y_true, y_pred, embeddings)
    else:
        return calculate_recognition_metrics(y_true, y_pred)

