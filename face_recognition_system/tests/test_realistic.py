#!/usr/bin/env python3
"""
使用模拟深度学习特征测试DGFC算法

真实的人脸识别系统使用深度学习模型（如FaceNet/ArcFace）提取512维特征向量。
这些特征具有以下特点：
1. 同一个人的特征向量余弦相似度很高 (>0.8)
2. 不同人的特征向量余弦相似度较低 (<0.4)
3. 特征经过L2归一化

本测试模拟这种真实的特征分布来评估DGFC算法
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.clustering_engine import DGFCClusteringEngine, DGFCConfig


def generate_realistic_embeddings(
    n_persons: int = 50,
    faces_per_person: tuple = (5, 30),
    embedding_dim: int = 512,
    intra_class_std: float = 0.15,  # 类内标准差 (越小，同一人特征越相似)
    seed: int = 42,
):
    """
    生成模拟的真实人脸特征
    
    模拟深度学习模型输出的特征分布:
    - 每个人有一个中心特征向量
    - 同一人的不同照片围绕中心有小幅变化
    - 不同人的中心特征向量相互正交（理想情况）
    
    Args:
        n_persons: 人数
        faces_per_person: 每个人的人脸数范围 (min, max)
        embedding_dim: 特征维度
        intra_class_std: 类内标准差 (控制同一人特征的变化程度)
        seed: 随机种子
        
    Returns:
        embeddings: 特征矩阵 (N, D)
        labels: 真实标签
        person_info: 每个人的信息
    """
    np.random.seed(seed)
    
    # 为每个人生成一个随机的中心特征向量
    person_centers = np.random.randn(n_persons, embedding_dim)
    person_centers = person_centers / np.linalg.norm(person_centers, axis=1, keepdims=True)
    
    embeddings = []
    labels = []
    person_info = []
    
    for person_id in range(n_persons):
        # 随机决定这个人有多少张照片
        n_faces = np.random.randint(faces_per_person[0], faces_per_person[1] + 1)
        
        # 生成围绕中心的变化
        noise = np.random.randn(n_faces, embedding_dim) * intra_class_std
        person_embeddings = person_centers[person_id] + noise
        
        # L2归一化
        person_embeddings = person_embeddings / np.linalg.norm(
            person_embeddings, axis=1, keepdims=True
        )
        
        embeddings.append(person_embeddings)
        labels.extend([person_id] * n_faces)
        person_info.append({
            "person_id": person_id,
            "n_faces": n_faces,
        })
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels, person_info


def generate_challenging_embeddings(
    n_persons: int = 50,
    faces_per_person: tuple = (5, 30),
    embedding_dim: int = 512,
    noise_ratio: float = 0.05,  # 噪声点比例
    seed: int = 42,
):
    """
    生成有挑战性的人脸特征（变密度 + 噪声）
    
    特点：
    1. 不同身份的簇有不同的密度（类内方差不同）
    2. 添加随机噪声点
    3. 更真实地模拟人脸识别场景中的困难情况
    """
    np.random.seed(seed)
    
    person_centers = np.random.randn(n_persons, embedding_dim)
    person_centers = person_centers / np.linalg.norm(person_centers, axis=1, keepdims=True)
    
    embeddings = []
    labels = []
    person_info = []
    
    for person_id in range(n_persons):
        n_faces = np.random.randint(faces_per_person[0], faces_per_person[1] + 1)
        
        # 【关键】不同身份使用不同的类内方差 (0.02 ~ 0.06)
        # 这模拟了真实场景：有些人的照片差异大，有些人差异小
        # 变化范围3倍，足以展示DGFC对变密度的适应性
        intra_std = np.random.uniform(0.02, 0.06)
        
        noise = np.random.randn(n_faces, embedding_dim) * intra_std
        person_embeddings = person_centers[person_id] + noise
        person_embeddings = person_embeddings / np.linalg.norm(
            person_embeddings, axis=1, keepdims=True
        )
        
        embeddings.append(person_embeddings)
        labels.extend([person_id] * n_faces)
        person_info.append({
            "person_id": person_id,
            "n_faces": n_faces,
            "intra_std": intra_std,
        })
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # 【关键】添加噪声点（模拟误检或未知身份）
    n_noise = int(len(labels) * noise_ratio)
    if n_noise > 0:
        noise_embeddings = np.random.randn(n_noise, embedding_dim)
        noise_embeddings = noise_embeddings / np.linalg.norm(noise_embeddings, axis=1, keepdims=True)
        noise_labels = np.full(n_noise, -1)  # 噪声标签为-1
        
        embeddings = np.vstack([embeddings, noise_embeddings])
        labels = np.concatenate([labels, noise_labels])
        
        # 打乱顺序
        perm = np.random.permutation(len(labels))
        embeddings = embeddings[perm]
        labels = labels[perm]
    
    return embeddings, labels, person_info


def analyze_feature_distribution(embeddings, labels):
    """分析特征分布"""
    print("\n特征分布分析:")
    print("-" * 40)
    
    unique_labels = np.unique(labels)
    
    # 计算类内相似度
    intra_sims = []
    for label in unique_labels[:10]:  # 采样10个人
        mask = labels == label
        class_embeddings = embeddings[mask]
        if len(class_embeddings) > 1:
            # 计算类内所有对的余弦相似度
            sims = np.dot(class_embeddings, class_embeddings.T)
            # 取上三角（排除对角线）
            triu_indices = np.triu_indices(len(class_embeddings), k=1)
            intra_sims.extend(sims[triu_indices].tolist())
    
    # 计算类间相似度
    inter_sims = []
    for i in range(min(100, len(unique_labels))):
        for j in range(i+1, min(100, len(unique_labels))):
            mask_i = labels == unique_labels[i]
            mask_j = labels == unique_labels[j]
            emb_i = embeddings[mask_i][0]
            emb_j = embeddings[mask_j][0]
            inter_sims.append(np.dot(emb_i, emb_j))
    
    print(f"类内相似度: {np.mean(intra_sims):.4f} ± {np.std(intra_sims):.4f}")
    print(f"类间相似度: {np.mean(inter_sims):.4f} ± {np.std(inter_sims):.4f}")
    print(f"可分性 (类内-类间): {np.mean(intra_sims) - np.mean(inter_sims):.4f}")


def evaluate_clustering(y_true, y_pred, exclude_noise=True):
    """评估聚类结果（支持噪声标签）"""
    from sklearn.metrics import (
        adjusted_rand_score, 
        normalized_mutual_info_score,
        homogeneity_score,
        completeness_score,
        v_measure_score,
    )
    
    if exclude_noise:
        # 排除噪声点（标签为-1）进行评估
        mask = y_true >= 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-measure": v_measure_score(y_true, y_pred),
    }


def test_dgfc(embeddings, labels, knn_k=30):
    """测试DGFC-v2算法"""
    n_samples = len(labels)
    
    config = DGFCConfig(
        knn_k=min(knn_k, n_samples - 1),
        metric="cosine",
        density_percentile=0.7,       # 前70%密度被视为高密度
        delta_percentile=0.9,         # 用于γ值筛选
        min_cluster_size=5,
        radius_alpha=1.0,
        merge_similarity_threshold=0.5,  # 簇均值相似度>0.5才合并
        merge_overlap_ratio=0.2,         # 边界重叠>20%才考虑合并
        assign_k=min(30, n_samples - 1),
        verbose=True,
    )
    
    engine = DGFCClusteringEngine(config)
    
    print(f"\n运行DGFC-v2 (knn_k={knn_k})...")
    start = time.time()
    result = engine.fit_predict(embeddings)
    elapsed = time.time() - start
    
    return result, elapsed


def compare_algorithms(embeddings, labels):
    """与其他算法对比"""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    
    n_clusters = len(np.unique(labels))
    n_samples = len(labels)
    
    print("\n" + "="*60)
    print("算法对比")
    print("="*60)
    
    results = {}
    
    # K-Means (需要知道簇数)
    print("\n[1] K-Means (已知簇数)...")
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(embeddings)
    t_kmeans = time.time() - start
    results["K-Means"] = (evaluate_clustering(labels, y_kmeans), n_clusters, t_kmeans)
    
    # DBSCAN
    print("[2] DBSCAN (自动发现簇数)...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    eps = np.percentile(distances[:, -1], 85)
    
    start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=3, metric='cosine')
    y_dbscan = dbscan.fit_predict(embeddings)
    t_dbscan = time.time() - start
    n_dbscan = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
    results["DBSCAN"] = (evaluate_clustering(labels, y_dbscan), n_dbscan, t_dbscan)
    
    # 层次聚类 (需要知道簇数)
    print("[3] Agglomerative (已知簇数)...")
    start = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    y_agg = agg.fit_predict(embeddings)
    t_agg = time.time() - start
    results["Agglomerative"] = (evaluate_clustering(labels, y_agg), n_clusters, t_agg)
    
    # DGFC-v2
    print("[4] DGFC-v2 (自动发现簇数)...")
    start = time.time()
    config = DGFCConfig(
        knn_k=min(30, n_samples - 1),
        metric="cosine",
        density_percentile=0.7,
        merge_similarity_threshold=0.5,
        merge_overlap_ratio=0.2,
        verbose=False,
    )
    engine = DGFCClusteringEngine(config)
    dgfc_result = engine.fit_predict(embeddings)
    t_dgfc = time.time() - start
    results["DGFC-v2"] = (
        evaluate_clustering(labels, dgfc_result.labels), 
        dgfc_result.n_clusters, 
        t_dgfc
    )
    
    # 打印结果
    print("\n" + "-"*70)
    print(f"{'算法':<15} {'ARI':>8} {'NMI':>8} {'簇数':>6} {'耗时(s)':>8} {'需要k?':>8}")
    print("-"*70)
    
    for name, (metrics, n_c, t) in results.items():
        needs_k = "是" if name in ["K-Means", "Agglomerative"] else "否"
        print(f"{name:<15} {metrics['ARI']:>8.4f} {metrics['NMI']:>8.4f} {n_c:>6} {t:>8.3f} {needs_k:>8}")
    
    print("-"*70)
    print(f"{'真实':<15} {'-':>8} {'-':>8} {n_clusters:>6}")
    
    # 找出最佳（自动发现簇数的算法）
    auto_algos = {k: v for k, v in results.items() if k in ["DBSCAN", "DGFC-v2"]}
    if auto_algos:
        best = max(auto_algos.items(), key=lambda x: x[1][0]['ARI'])
        print(f"\n最佳(自动发现簇数): {best[0]} (ARI={best[1][0]['ARI']:.4f})")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="使用模拟深度学习特征测试DGFC")
    parser.add_argument("--n-persons", type=int, default=50, help="人数")
    parser.add_argument("--min-faces", type=int, default=5, help="每人最少人脸数")
    parser.add_argument("--max-faces", type=int, default=30, help="每人最多人脸数")
    parser.add_argument("--intra-std", type=float, default=0.15, help="类内标准差")
    parser.add_argument("--compare", action="store_true", help="与其他算法对比")
    parser.add_argument("--challenging", action="store_true", help="使用挑战性数据集(变密度+噪声)")
    parser.add_argument("--noise-ratio", type=float, default=0.05, help="噪声点比例")
    
    args = parser.parse_args()
    
    print("\n" + "█"*60)
    print("  DGFC人脸识别系统 - 模拟深度学习特征测试")
    print("█"*60)
    
    # 生成数据
    if args.challenging:
        print(f"\n生成挑战性数据: {args.n_persons}人, 每人{args.min_faces}-{args.max_faces}张")
        print(f"特点: 变密度簇 + {args.noise_ratio*100:.0f}%噪声")
        embeddings, labels, person_info = generate_challenging_embeddings(
            n_persons=args.n_persons,
            faces_per_person=(args.min_faces, args.max_faces),
            noise_ratio=args.noise_ratio,
        )
    else:
        print(f"\n生成模拟数据: {args.n_persons}人, 每人{args.min_faces}-{args.max_faces}张")
        embeddings, labels, person_info = generate_realistic_embeddings(
            n_persons=args.n_persons,
            faces_per_person=(args.min_faces, args.max_faces),
            intra_class_std=args.intra_std,
        )
    
    n_samples = len(labels)
    # 对于有噪声的数据，只计算非噪声点的类别数
    non_noise_labels = labels[labels >= 0]
    n_classes = len(np.unique(non_noise_labels))
    n_noise = np.sum(labels == -1)
    
    if n_noise > 0:
        print(f"生成完成: {n_samples}张人脸, {n_classes}个身份, {n_noise}个噪声点")
    else:
        print(f"生成完成: {n_samples}张人脸, {n_classes}个身份")
    
    # 分析特征分布
    analyze_feature_distribution(embeddings, labels)
    
    # 测试DGFC
    print("\n" + "="*60)
    print("DGFC聚类测试")
    print("="*60)
    
    result, elapsed = test_dgfc(embeddings, labels)
    
    print(f"\n聚类完成，耗时: {elapsed:.3f}秒")
    print(f"发现簇数: {result.n_clusters} (真实: {n_classes})")
    print(f"核心点: {np.sum(result.core_mask)} / {n_samples}")
    
    # 评估
    metrics = evaluate_clustering(labels, result.labels)
    
    print("\n评估指标:")
    for name, value in metrics.items():
        bar = "█" * int(value * 30) + "░" * (30 - int(value * 30))
        print(f"  {name:<12}: {value:.4f} |{bar}|")
    
    # 评级
    ari = metrics["ARI"]
    if ari >= 0.9:
        grade = "★★★★★ 卓越"
    elif ari >= 0.8:
        grade = "★★★★☆ 优秀"
    elif ari >= 0.6:
        grade = "★★★☆☆ 良好"
    elif ari >= 0.4:
        grade = "★★☆☆☆ 中等"
    else:
        grade = "★☆☆☆☆ 待改进"
    
    print(f"\n综合评价: {grade}")
    
    # 对比
    if args.compare:
        compare_algorithms(embeddings, labels)
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

