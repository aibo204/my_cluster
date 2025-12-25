#!/usr/bin/env python3
"""
在LFW (Labeled Faces in the Wild) 数据集上测试DGFC聚类算法

LFW是一个广泛使用的人脸识别基准数据集，包含5749个人的13233张人脸图像。
"""
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.clustering_engine import DGFCClusteringEngine, DGFCConfig


def load_lfw_data(min_faces_per_person=10, use_pca=True, pca_dim=128):
    """
    加载LFW数据集
    
    Args:
        min_faces_per_person: 每个人最少的人脸数量 (过滤低样本的人)
        use_pca: 是否使用PCA降维 (加速处理)
        pca_dim: PCA目标维度
    """
    print("\n" + "="*60)
    print("加载LFW数据集...")
    print("="*60)
    
    try:
        from sklearn.datasets import fetch_lfw_people
        from sklearn.decomposition import PCA
    except ImportError:
        print("需要安装 scikit-learn: pip install scikit-learn")
        return None, None
    
    # 加载数据
    print(f"参数: min_faces_per_person={min_faces_per_person}")
    
    try:
        lfw = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=0.4,  # 缩小图像加速
            color=False,
        )
    except Exception as e:
        print(f"加载失败: {e}")
        print("首次运行需要下载数据，请确保网络连接正常")
        return None, None
    
    X = lfw.data.astype(np.float32)
    y = lfw.target
    target_names = lfw.target_names
    
    print(f"原始数据: {X.shape[0]} 张图像, {len(target_names)} 个人")
    print(f"图像尺寸: {lfw.images.shape[1]}x{lfw.images.shape[2]}")
    print(f"特征维度: {X.shape[1]}")
    
    # 样本分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"样本分布: 最少{counts.min()}张, 最多{counts.max()}张, 平均{counts.mean():.1f}张")
    
    # PCA降维
    if use_pca:
        print(f"\n使用PCA降维到 {pca_dim} 维...")
        pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
        X_pca = pca.fit_transform(X)
        print(f"降维后维度: {X_pca.shape}")
        
        # L2归一化
        X_pca = X_pca / np.linalg.norm(X_pca, axis=1, keepdims=True)
        return X_pca, y
    
    return X, y


def evaluate_clustering(y_true, y_pred):
    """评估聚类结果"""
    from sklearn.metrics import (
        adjusted_rand_score, 
        normalized_mutual_info_score,
        homogeneity_score,
        completeness_score,
        v_measure_score,
    )
    
    metrics = {
        "ARI (调整兰德指数)": adjusted_rand_score(y_true, y_pred),
        "NMI (归一化互信息)": normalized_mutual_info_score(y_true, y_pred),
        "同质性 (Homogeneity)": homogeneity_score(y_true, y_pred),
        "完整性 (Completeness)": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
    }
    
    return metrics


def test_dgfc_on_lfw(
    min_faces_per_person=20,
    pca_dim=128,
    knn_k=30,
    tau=0.85,
):
    """
    在LFW数据集上测试DGFC算法
    """
    # 加载数据
    X, y_true = load_lfw_data(
        min_faces_per_person=min_faces_per_person,
        use_pca=True,
        pca_dim=pca_dim,
    )
    
    if X is None:
        return
    
    n_samples = len(y_true)
    n_classes = len(np.unique(y_true))
    
    print("\n" + "="*60)
    print("运行DGFC聚类算法")
    print("="*60)
    print(f"数据规模: {n_samples} 张人脸, {n_classes} 个真实身份")
    print(f"配置: knn_k={knn_k}, tau={tau}")
    
    # 配置DGFC
    config = DGFCConfig(
        knn_k=min(knn_k, n_samples - 1),
        metric="cosine",
        tau=tau,
        use_mahalanobis=True,
        mahalanobis_q=0.90,
        radius_alpha=1.0,
        assign_k=min(20, n_samples - 1),
        verbose=True,
    )
    
    # 运行聚类
    engine = DGFCClusteringEngine(config)
    
    print("\n开始聚类...")
    start_time = time.time()
    result = engine.fit_predict(X)
    elapsed = time.time() - start_time
    
    print(f"\n聚类完成，耗时: {elapsed:.2f} 秒")
    
    # 结果分析
    print("\n" + "="*60)
    print("聚类结果分析")
    print("="*60)
    
    y_pred = result.labels
    n_clusters = result.n_clusters
    n_core = np.sum(result.core_mask)
    n_unassigned = np.sum(y_pred == 0)
    
    print(f"发现簇数量: {n_clusters} (真实: {n_classes})")
    print(f"核心点: {n_core} / {n_samples} ({100*n_core/n_samples:.1f}%)")
    print(f"未分配: {n_unassigned} ({100*n_unassigned/n_samples:.1f}%)")
    
    # 簇大小分布
    if result.cluster_indices:
        sizes = [len(indices) for indices in result.cluster_indices.values()]
        print(f"簇大小: 最小={min(sizes)}, 最大={max(sizes)}, 平均={np.mean(sizes):.1f}")
    
    # 评估指标
    print("\n" + "="*60)
    print("评估指标")
    print("="*60)
    
    metrics = evaluate_clustering(y_true, y_pred)
    for name, value in metrics.items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"{name}: {value:.4f} |{bar}|")
    
    # 判断效果
    ari = metrics["ARI (调整兰德指数)"]
    if ari >= 0.7:
        grade = "★★★ 优秀"
    elif ari >= 0.5:
        grade = "★★☆ 良好"
    elif ari >= 0.3:
        grade = "★☆☆ 一般"
    else:
        grade = "☆☆☆ 待改进"
    
    print(f"\n综合评价: {grade}")
    
    return result, metrics


def compare_with_baselines(X, y_true):
    """
    与其他聚类算法对比
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    n_clusters = len(np.unique(y_true))
    
    print("\n" + "="*60)
    print("与基线算法对比")
    print("="*60)
    
    results = {}
    
    # K-Means
    print("\n[1] K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    results["K-Means"] = {
        "ARI": adjusted_rand_score(y_true, y_kmeans),
        "NMI": normalized_mutual_info_score(y_true, y_kmeans),
    }
    
    # DBSCAN
    print("[2] DBSCAN...")
    # 自动搜索最佳eps
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    eps = np.percentile(distances[:, -1], 90)
    
    dbscan = DBSCAN(eps=eps, min_samples=3, metric='cosine')
    y_dbscan = dbscan.fit_predict(X)
    results["DBSCAN"] = {
        "ARI": adjusted_rand_score(y_true, y_dbscan),
        "NMI": normalized_mutual_info_score(y_true, y_dbscan),
        "n_clusters": len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0),
    }
    
    # 层次聚类
    print("[3] 层次聚类 (Agglomerative)...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_agg = agg.fit_predict(X)
    results["Agglomerative"] = {
        "ARI": adjusted_rand_score(y_true, y_agg),
        "NMI": normalized_mutual_info_score(y_true, y_agg),
    }
    
    # DGFC
    print("[4] DGFC (本文算法)...")
    config = DGFCConfig(knn_k=min(30, len(X)-1), tau=0.85, verbose=False)
    engine = DGFCClusteringEngine(config)
    dgfc_result = engine.fit_predict(X)
    results["DGFC"] = {
        "ARI": adjusted_rand_score(y_true, dgfc_result.labels),
        "NMI": normalized_mutual_info_score(y_true, dgfc_result.labels),
        "n_clusters": dgfc_result.n_clusters,
    }
    
    # 打印对比表格
    print("\n" + "-"*50)
    print(f"{'算法':<15} {'ARI':>10} {'NMI':>10} {'簇数':>8}")
    print("-"*50)
    
    for name, m in results.items():
        n_c = m.get('n_clusters', n_clusters)
        print(f"{name:<15} {m['ARI']:>10.4f} {m['NMI']:>10.4f} {n_c:>8}")
    
    print("-"*50)
    print(f"{'真实':<15} {'-':>10} {'-':>10} {n_clusters:>8}")
    
    # 找出最佳
    best_ari = max(results.items(), key=lambda x: x[1]['ARI'])
    print(f"\n最佳ARI: {best_ari[0]} ({best_ari[1]['ARI']:.4f})")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="在LFW数据集上测试DGFC算法")
    parser.add_argument("--min-faces", type=int, default=20, 
                       help="每个人最少的人脸数 (默认20)")
    parser.add_argument("--pca-dim", type=int, default=128,
                       help="PCA降维维度 (默认128)")
    parser.add_argument("--knn-k", type=int, default=30,
                       help="kNN的k值 (默认30)")
    parser.add_argument("--tau", type=float, default=0.85,
                       help="核心点阈值 (默认0.85)")
    parser.add_argument("--compare", action="store_true",
                       help="与其他算法对比")
    
    args = parser.parse_args()
    
    print("\n" + "█"*60)
    print("  DGFC人脸识别系统 - LFW数据集测试")
    print("█"*60)
    
    # 测试DGFC
    result, metrics = test_dgfc_on_lfw(
        min_faces_per_person=args.min_faces,
        pca_dim=args.pca_dim,
        knn_k=args.knn_k,
        tau=args.tau,
    )
    
    # 对比测试
    if args.compare:
        X, y = load_lfw_data(args.min_faces, True, args.pca_dim)
        if X is not None:
            compare_with_baselines(X, y)
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

