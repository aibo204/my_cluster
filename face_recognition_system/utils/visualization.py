"""
可视化工具
Visualization Utilities for DGFC Face Recognition System
"""
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, visualization disabled")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def visualize_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    title: str = "Face Clustering Visualization",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    可视化聚类结果
    
    Args:
        embeddings: 特征向量矩阵 (N, D)
        labels: 聚类标签 (N,)
        method: 降维方法 ("tsne" or "pca")
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure对象
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is required for visualization")
        return None
    
    if not SKLEARN_AVAILABLE:
        logger.error("Scikit-learn is required for dimensionality reduction")
        return None
    
    # 降维
    logger.info(f"Reducing dimensions using {method.upper()}...")
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels > 0])
    
    # 颜色映射
    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))
    
    for label in unique_labels:
        mask = labels == label
        
        if label == 0:
            # 未分配的点 (噪声)
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c='gray',
                marker='x',
                s=30,
                alpha=0.5,
                label='Unassigned'
            )
        else:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(label - 1)],
                marker='o',
                s=50,
                alpha=0.7,
                label=f'Cluster {label}'
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    
    # 限制图例数量
    if n_clusters <= 10:
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_density_features(
    gsdata: np.ndarray,
    labels: np.ndarray,
    core_mask: Optional[np.ndarray] = None,
    title: str = "Density Feature Space",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    可视化密度特征空间
    
    Args:
        gsdata: 密度特征数据 (N, 2)
        labels: 聚类标签
        core_mask: 核心点掩码
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        show: 是否显示
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is required for visualization")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制所有点
    scatter = ax.scatter(
        gsdata[:, 0],
        gsdata[:, 1],
        c=labels,
        cmap='turbo',
        s=30,
        alpha=0.6,
    )
    
    # 高亮核心点
    if core_mask is not None:
        ax.scatter(
            gsdata[core_mask, 0],
            gsdata[core_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=100,
            linewidths=1.5,
            label='Core Points'
        )
    
    ax.set_xlabel('Normalized Inverse Variance', fontsize=11)
    ax.set_ylabel('Normalized Inverse Median Distance', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    
    if core_mask is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    绘制混淆矩阵
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def draw_face_grid(
    face_images: List[np.ndarray],
    labels: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None,
    max_images: int = 25,
    cols: int = 5,
    figsize: tuple = (15, 12),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    绘制人脸图像网格
    
    Args:
        face_images: 人脸图像列表
        labels: 聚类/识别标签
        names: 人员姓名
        max_images: 最大显示数量
        cols: 每行列数
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    import cv2
    
    n_images = min(len(face_images), max_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n_images:
            img = face_images[i]
            # BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            
            title = ""
            if labels is not None and i < len(labels):
                title = f"Cluster {labels[i]}"
            if names is not None and i < len(names):
                title = names[i]
            
            ax.set_title(title, fontsize=9)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

