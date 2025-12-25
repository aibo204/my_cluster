"""
DGFC-v2: Density-Guided Flood Clustering (改进版)

核心创新点:
1. 局部密度估计 - 基于k-NN距离的局部密度,避免全局GMM的缺陷
2. 密度峰值检测 - 结合ρ(密度)和δ(最近高密度点距离)自动发现簇中心
3. 洪泛填充聚类 - 从密度峰值向外扩展,使用自适应半径
4. 簇间相似性验证 - 合并前验证两个簇的特征分布是否兼容

相比原始DGFC的改进:
- 用局部密度排名代替全局GMM,更好地适应变密度数据
- 引入δ值识别真正的簇中心,而非仅靠密度
- 合并决策考虑簇内特征一致性,防止误合并
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _minmax_normalize(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """最大最小归一化"""
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.full_like(x, lo, dtype=float)
    z = (x - mn) / (mx - mn)
    return z * (hi - lo) + lo


class DSU:
    """并查集 (Disjoint Set Union)"""
    
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)
    
    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root
    
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class DGFCConfig:
    """DGFC-v2聚类配置"""
    # kNN参数
    knn_k: int = 30
    metric: str = "cosine"
    
    # 密度峰值检测参数
    density_percentile: float = 0.8  # 密度前x%被视为高密度
    delta_percentile: float = 0.9    # δ值前x%被视为簇中心候选
    min_cluster_size: int = 5        # 最小簇大小
    
    # 自适应半径
    radius_alpha: float = 1.0
    radius_clip_quantiles: Tuple[float, float] = (0.05, 0.95)
    
    # 合并验证参数
    merge_similarity_threshold: float = 0.6  # 簇间平均相似度阈值
    merge_overlap_ratio: float = 0.3         # 边界重叠比例阈值
    
    # 分配参数
    assign_k: int = 20
    
    # 其他
    eps: float = 1e-12
    l2_normalize: bool = True
    random_state: int = 42
    verbose: bool = True
    n_jobs: int = -1


@dataclass
class ClusteringResult:
    """聚类结果"""
    labels: np.ndarray
    n_clusters: int
    core_mask: np.ndarray
    cluster_indices: Dict[int, List[int]] = field(default_factory=dict)
    cluster_centers: Dict[int, np.ndarray] = field(default_factory=dict)
    density_scores: np.ndarray = None
    delta_scores: np.ndarray = None
    peak_indices: np.ndarray = None
    extras: Dict[str, Any] = field(default_factory=dict)


class DGFCClusteringEngine:
    """
    DGFC-v2 聚类引擎
    
    算法流程:
    1. 构建kNN图
    2. 计算局部密度ρ (基于k-NN距离的倒数)
    3. 计算δ值 (到最近更高密度点的距离)
    4. 识别密度峰值 (高ρ + 高δ)
    5. 从峰值开始洪泛填充
    6. 验证并合并相似簇
    7. 分配剩余点
    """
    
    def __init__(self, config: Optional[DGFCConfig] = None):
        self.config = config or DGFCConfig()
        self._knn_index = None
    
    def _log(self, msg: str):
        if self.config.verbose:
            print(f"[DGFC-v2] {msg}")
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        """执行聚类"""
        cfg = self.config
        n = X.shape[0]
        
        if cfg.l2_normalize:
            X = self._l2_normalize(X)
        
        # Step 1: 构建kNN图
        self._log("Step 1: 构建kNN图...")
        knn_dist, knn_ind = self._build_knn_graph(X)
        
        # Step 2: 计算局部密度ρ
        self._log("Step 2: 计算局部密度...")
        rho = self._compute_local_density(knn_dist)
        
        # Step 3: 计算δ值
        self._log("Step 3: 计算δ值 (最近高密度点距离)...")
        delta, nearest_higher = self._compute_delta(X, rho)
        
        # Step 4: 识别密度峰值
        self._log("Step 4: 识别密度峰值...")
        peak_mask, peak_indices = self._detect_peaks(rho, delta)
        n_peaks = len(peak_indices)
        self._log(f"   发现 {n_peaks} 个密度峰值")
        
        # Step 5: 洪泛填充聚类
        self._log("Step 5: 从峰值开始洪泛填充...")
        radii = self._compute_adaptive_radii(knn_dist, rho)
        labels, dsu = self._flood_fill_from_peaks(
            X, knn_ind, knn_dist, rho, radii, peak_indices, nearest_higher
        )
        
        # Step 6: 合并相似簇
        self._log("Step 6: 验证并合并相似簇...")
        labels = self._merge_similar_clusters(X, labels, knn_ind, knn_dist)
        
        # Step 7: 分配剩余点
        self._log("Step 7: 分配剩余点...")
        labels = self._assign_remaining(X, labels, knn_ind, knn_dist)
        
        # 计算簇信息
        n_clusters = len(np.unique(labels[labels > 0]))
        cluster_indices = self._get_cluster_indices(labels)
        cluster_centers = self._compute_cluster_centers(X, labels)
        
        self._log(f"聚类完成: 发现 {n_clusters} 个簇")
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            core_mask=peak_mask,
            cluster_indices=cluster_indices,
            cluster_centers=cluster_centers,
            density_scores=rho,
            delta_scores=delta,
            peak_indices=peak_indices,
            extras={
                "knn_indices": knn_ind,
                "knn_distances": knn_dist,
                "radii": radii,
                "nearest_higher": nearest_higher,
            }
        )
    
    def _l2_normalize(self, X: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + self.config.eps)
    
    def _build_knn_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """构建kNN图"""
        cfg = self.config
        k = min(cfg.knn_k, X.shape[0] - 1)
        
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # +1 因为包含自身
            metric=cfg.metric,
            n_jobs=cfg.n_jobs,
            algorithm="auto",
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        self._knn_index = nn
        
        # 排除自身
        return distances[:, 1:], indices[:, 1:]
    
    def _compute_local_density(self, knn_dist: np.ndarray) -> np.ndarray:
        """
        计算局部密度 ρ
        
        使用k-NN距离的倒数作为密度估计:
        ρ_i = 1 / (mean(d_ij) + eps)
        
        这比全局GMM更能适应局部密度变化
        """
        cfg = self.config
        
        # 使用平均k-NN距离的倒数
        mean_dist = np.mean(knn_dist, axis=1)
        rho = 1.0 / (mean_dist + cfg.eps)
        
        # 归一化到[0,1]
        rho = _minmax_normalize(rho)
        
        return rho
    
    def _compute_delta(
        self, X: np.ndarray, rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 δ 值 (到最近更高密度点的距离)
        
        对于每个点i:
        δ_i = min{d(i,j) : ρ_j > ρ_i}
        
        δ值高意味着该点是局部密度峰值(周围没有更高密度的点)
        这是识别簇中心的关键
        """
        cfg = self.config
        n = X.shape[0]
        
        delta = np.zeros(n)
        nearest_higher = np.full(n, -1, dtype=int)
        
        # 按密度降序排列
        order = np.argsort(rho)[::-1]
        
        # 计算距离矩阵(如果数据量大,可以使用近似方法)
        if cfg.metric == "cosine":
            # 余弦距离 = 1 - 余弦相似度
            sim_matrix = np.dot(X, X.T)
            dist_matrix = 1.0 - sim_matrix
        else:
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(X, X, metric="euclidean")
        
        # 对于密度最高的点,δ设为最大距离
        delta[order[0]] = np.max(dist_matrix[order[0]])
        nearest_higher[order[0]] = order[0]  # 指向自己
        
        # 对于其他点,找最近的更高密度点
        for i in range(1, n):
            idx = order[i]
            higher_density_points = order[:i]  # 前面的点密度都更高
            
            distances_to_higher = dist_matrix[idx, higher_density_points]
            min_idx = np.argmin(distances_to_higher)
            
            delta[idx] = distances_to_higher[min_idx]
            nearest_higher[idx] = higher_density_points[min_idx]
        
        return delta, nearest_higher
    
    def _detect_peaks(
        self, rho: np.ndarray, delta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测密度峰值（改进版）
        
        使用更鲁棒的峰值检测策略:
        1. 计算 γ = ρ * δ_normalized
        2. 使用统计方法找异常值(即真正的峰值)
        3. 结合密度约束
        """
        cfg = self.config
        n = len(rho)
        
        # 归一化δ
        delta_norm = _minmax_normalize(delta)
        
        # 综合评分 γ = ρ * δ
        gamma = rho * delta_norm
        
        # 方法: 基于γ值的统计分布找峰值
        # 使用 γ > mean + k*std 作为阈值
        gamma_mean = np.mean(gamma)
        gamma_std = np.std(gamma)
        
        # 自适应k值: 根据数据规模调整
        # 数据越多,k越小(允许更多峰值)
        k = max(0.5, 2.0 - np.log10(n) * 0.3)
        
        gamma_threshold = gamma_mean + k * gamma_std
        
        # 峰值候选: γ > threshold
        peak_candidates = gamma >= gamma_threshold
        
        # 额外条件: δ值也要足够高 (说明确实是局部最大)
        delta_threshold = np.percentile(delta_norm, 50)  # 中位数以上
        peak_candidates = peak_candidates & (delta_norm >= delta_threshold)
        
        # 如果候选太少,降低阈值
        min_peaks = max(2, int(np.sqrt(n) / 5))  # 至少sqrt(n)/5个峰值
        
        if np.sum(peak_candidates) < min_peaks:
            # 使用γ的分位数
            gamma_threshold = np.percentile(gamma, 95)
            peak_candidates = gamma >= gamma_threshold
        
        # 如果候选太多(超过sqrt(n)),收紧阈值
        max_peaks = int(np.sqrt(n) * 2)
        if np.sum(peak_candidates) > max_peaks:
            # 只取top-k
            top_k_idx = np.argsort(gamma)[-max_peaks:]
            peak_candidates = np.zeros(n, dtype=bool)
            peak_candidates[top_k_idx] = True
        
        # 确保至少有1个峰值
        if not np.any(peak_candidates):
            peak_candidates[np.argmax(gamma)] = True
        
        peak_indices = np.where(peak_candidates)[0]
        
        return peak_candidates, peak_indices
    
    def _compute_adaptive_radii(
        self, knn_dist: np.ndarray, rho: np.ndarray
    ) -> np.ndarray:
        """
        计算自适应半径
        
        高密度区域使用小半径,低密度区域使用大半径
        """
        cfg = self.config
        
        # 使用k-NN距离的中位数作为基础半径
        median_dist = np.median(knn_dist, axis=1)
        
        # 密度自适应: 高密度区用小半径,避免过度扩展
        # r_i = median_dist_i * (1 - 0.3 * rho_i)
        density_factor = 1.0 - 0.3 * rho
        r = cfg.radius_alpha * median_dist * density_factor
        
        # 裁剪极端值
        lo = float(np.quantile(r, cfg.radius_clip_quantiles[0]))
        hi = float(np.quantile(r, cfg.radius_clip_quantiles[1]))
        return np.clip(r, lo, hi)
    
    def _flood_fill_from_peaks(
        self,
        X: np.ndarray,
        knn_ind: np.ndarray,
        knn_dist: np.ndarray,
        rho: np.ndarray,
        radii: np.ndarray,
        peak_indices: np.ndarray,
        nearest_higher: np.ndarray,
    ) -> Tuple[np.ndarray, DSU]:
        """
        从密度峰值开始洪泛填充
        
        关键改进:
        1. 只从真正的密度峰值开始扩展
        2. 扩展时优先选择密度相近的邻居
        3. 使用密度梯度指导扩展方向
        """
        cfg = self.config
        n = X.shape[0]
        
        labels = np.zeros(n, dtype=int)
        dsu = DSU(n + len(peak_indices) + 1)
        
        # 为每个峰值分配一个簇ID
        for cluster_id, peak_idx in enumerate(peak_indices, start=1):
            labels[peak_idx] = cluster_id
        
        # 按密度降序处理所有点
        order = np.argsort(rho)[::-1]
        
        # 记录每个簇的边界点(用于后续合并验证)
        cluster_boundaries = defaultdict(set)
        
        for idx in order:
            if labels[idx] != 0:
                continue  # 已分配
            
            # 方法1: 沿着nearest_higher链向上追溯到峰值
            current = idx
            path = [current]
            
            while labels[current] == 0 and nearest_higher[current] != current:
                current = nearest_higher[current]
                if current in path:  # 防止死循环
                    break
                path.append(current)
            
            if labels[current] != 0:
                # 找到了有标签的点,分配相同标签
                cluster_label = labels[current]
                for p in path:
                    labels[p] = cluster_label
            else:
                # 没找到,使用kNN中最近的已标记点
                neighbors = knn_ind[idx]
                neighbor_dists = knn_dist[idx]
                
                for nb, dist in zip(neighbors, neighbor_dists):
                    if labels[nb] != 0 and dist <= radii[idx]:
                        labels[idx] = labels[nb]
                        break
        
        # 检测边界碰撞(不同簇相邻)
        collision_pairs = defaultdict(list)
        
        for i in range(n):
            if labels[i] == 0:
                continue
            
            for nb, dist in zip(knn_ind[i], knn_dist[i]):
                if labels[nb] != 0 and labels[nb] != labels[i]:
                    key = (min(labels[i], labels[nb]), max(labels[i], labels[nb]))
                    collision_pairs[key].append((i, nb, dist))
                    cluster_boundaries[labels[i]].add(i)
                    cluster_boundaries[labels[nb]].add(nb)
        
        return labels, dsu
    
    def _merge_similar_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        knn_ind: np.ndarray,
        knn_dist: np.ndarray,
    ) -> np.ndarray:
        """
        合并相似簇
        
        关键创新: 不是简单地碰撞即合并,而是验证两个簇的特征分布是否兼容
        
        合并条件:
        1. 边界区域有足够的重叠
        2. 两个簇的特征分布相似(均值接近)
        """
        cfg = self.config
        n = X.shape[0]
        
        # 获取所有簇
        unique_labels = np.unique(labels[labels > 0])
        if len(unique_labels) <= 1:
            return labels
        
        # 计算每个簇的统计信息
        cluster_stats = {}
        for c in unique_labels:
            mask = labels == c
            cluster_points = X[mask]
            cluster_stats[c] = {
                "mean": np.mean(cluster_points, axis=0),
                "size": np.sum(mask),
                "indices": np.where(mask)[0],
            }
        
        # 使用并查集管理合并
        dsu = DSU(int(np.max(unique_labels)) + 1)
        
        # 检测需要合并的簇对
        for i, c1 in enumerate(unique_labels):
            for c2 in unique_labels[i+1:]:
                # 计算两个簇的边界重叠
                indices1 = cluster_stats[c1]["indices"]
                indices2 = cluster_stats[c2]["indices"]
                
                # 边界点: 在kNN中有对方簇的点
                boundary1 = set()
                boundary2 = set()
                overlap_count = 0
                
                for idx in indices1:
                    for nb in knn_ind[idx]:
                        if labels[nb] == c2:
                            boundary1.add(idx)
                            overlap_count += 1
                            break
                
                for idx in indices2:
                    for nb in knn_ind[idx]:
                        if labels[nb] == c1:
                            boundary2.add(idx)
                            break
                
                if len(boundary1) == 0 or len(boundary2) == 0:
                    continue
                
                # 条件1: 边界重叠比例
                min_size = min(cluster_stats[c1]["size"], cluster_stats[c2]["size"])
                overlap_ratio = overlap_count / min_size
                
                # 条件2: 特征相似度(簇均值的余弦相似度)
                mean1 = cluster_stats[c1]["mean"]
                mean2 = cluster_stats[c2]["mean"]
                
                if cfg.metric == "cosine":
                    similarity = np.dot(mean1, mean2) / (
                        np.linalg.norm(mean1) * np.linalg.norm(mean2) + cfg.eps
                    )
                else:
                    # 欧氏距离转相似度
                    dist = np.linalg.norm(mean1 - mean2)
                    similarity = 1.0 / (1.0 + dist)
                
                # 决定是否合并
                should_merge = (
                    overlap_ratio >= cfg.merge_overlap_ratio and
                    similarity >= cfg.merge_similarity_threshold
                )
                
                if should_merge:
                    dsu.union(int(c1), int(c2))
        
        # 应用合并
        new_labels = labels.copy()
        for c in unique_labels:
            root = dsu.find(int(c))
            if root != c:
                new_labels[labels == c] = root
        
        # 重新编号为连续标签
        unique_new = np.unique(new_labels[new_labels > 0])
        label_map = {old: new for new, old in enumerate(unique_new, start=1)}
        
        for i in range(n):
            if new_labels[i] > 0:
                new_labels[i] = label_map[new_labels[i]]
        
        merged_count = len(unique_labels) - len(unique_new)
        if merged_count > 0:
            self._log(f"   合并了 {merged_count} 对相似簇")
        
        return new_labels
    
    def _assign_remaining(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        knn_ind: np.ndarray,
        knn_dist: np.ndarray,
    ) -> np.ndarray:
        """分配剩余未标记的点"""
        cfg = self.config
        lbl = labels.copy()
        unassigned = np.where(lbl == 0)[0]
        
        if len(unassigned) == 0:
            return lbl
        
        self._log(f"   {len(unassigned)} 个点待分配")
        
        k = min(cfg.assign_k, knn_ind.shape[1])
        
        # 迭代分配,直到所有点都被分配或无法继续
        max_iter = 10
        for iteration in range(max_iter):
            assigned_this_iter = 0
            
            for i in unassigned:
                if lbl[i] != 0:
                    continue
                
                neighbors = knn_ind[i, :k]
                distances = knn_dist[i, :k]
                neighbor_labels = lbl[neighbors]
                
                # 只考虑已分配的邻居
                mask = neighbor_labels > 0
                if not np.any(mask):
                    continue
                
                cand_labels = neighbor_labels[mask]
                cand_dists = distances[mask]
                
                # 距离加权投票
                weights = 1.0 / (cand_dists + cfg.eps)
                
                unique_labels = np.unique(cand_labels)
                best_label = int(unique_labels[0])
                best_weight = 0.0
                
                for ul in unique_labels:
                    w = np.sum(weights[cand_labels == ul])
                    if w > best_weight:
                        best_weight = w
                        best_label = int(ul)
                
                lbl[i] = best_label
                assigned_this_iter += 1
            
            if assigned_this_iter == 0:
                break
            
            unassigned = np.where(lbl == 0)[0]
        
        # 剩余无法分配的点标记为噪声(-1)
        remaining = np.sum(lbl == 0)
        if remaining > 0:
            lbl[lbl == 0] = -1
            self._log(f"   {remaining} 个点标记为噪声")
        
        return lbl
    
    def _get_cluster_indices(self, labels: np.ndarray) -> Dict[int, List[int]]:
        """获取每个簇的样本索引"""
        cluster_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label > 0:
                cluster_indices[int(label)].append(i)
        return dict(cluster_indices)
    
    def _compute_cluster_centers(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """计算簇中心"""
        centers = {}
        for label in np.unique(labels):
            if label > 0:
                mask = labels == label
                centers[int(label)] = np.mean(X[mask], axis=0)
        return centers


# 保持向后兼容
class ClusteringEngine(DGFCClusteringEngine):
    """兼容旧接口"""
    pass


# 导出
__all__ = [
    "DGFCClusteringEngine",
    "DGFCConfig",
    "ClusteringResult",
    "ClusteringEngine",
    "DSU",
]
