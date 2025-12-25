"""
系统配置文件
Face Recognition System Configuration
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class DetectionConfig:
    """人脸检测配置"""
    # 检测器类型: mtcnn, retinaface, opencv
    detector_type: Literal["mtcnn", "retinaface", "opencv"] = "mtcnn"
    
    # MTCNN 配置
    mtcnn_image_size: int = 160
    mtcnn_margin: int = 20
    mtcnn_min_face_size: int = 20
    mtcnn_thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    mtcnn_factor: float = 0.709
    
    # 置信度阈值
    confidence_threshold: float = 0.9
    
    # 最大检测人脸数
    max_faces: int = 50


@dataclass
class FeatureConfig:
    """特征提取配置"""
    # 模型类型: arcface, facenet, vggface
    model_type: Literal["arcface", "facenet", "vggface"] = "facenet"
    
    # 特征维度
    embedding_dim: int = 512
    
    # 输入尺寸
    input_size: Tuple[int, int] = (160, 160)
    
    # 是否L2归一化
    l2_normalize: bool = True
    
    # 预训练模型路径
    pretrained_path: Optional[str] = None
    
    # 设备
    device: str = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"


@dataclass  
class ClusteringConfig:
    """DGFC聚类算法配置"""
    # kNN参数
    knn_k: int = 50
    metric: Literal["euclidean", "cosine"] = "cosine"
    
    # GMM核心点选择
    gmm_components: int = 2
    gmm_n_init: int = 5
    tau: float = 0.85
    
    # Mahalanobis约束
    use_mahalanobis: bool = True
    mahalanobis_q: float = 0.95
    
    # 自适应半径
    radius_alpha: float = 1.0
    radius_clip_quantiles: Tuple[float, float] = (0.05, 0.95)
    
    # 分配参数
    assign_k: int = 30
    assign_mode: Literal["majority", "distance_weighted"] = "majority"
    
    # 其他
    random_state: int = 42
    verbose: bool = True


@dataclass
class DatabaseConfig:
    """数据库配置"""
    # SQLite数据库路径
    db_path: str = str(BASE_DIR / "data" / "faces.db")
    
    # 人脸图像存储路径
    face_images_dir: str = str(BASE_DIR / "data" / "faces")
    
    # 原始图片存储路径
    original_images_dir: str = str(BASE_DIR / "data" / "originals")
    
    # FAISS索引路径
    faiss_index_path: str = str(BASE_DIR / "data" / "faiss_index.bin")
    
    # 最大存储人脸数
    max_faces: int = 100000


@dataclass
class RecognitionConfig:
    """识别配置"""
    # 相似度阈值
    similarity_threshold: float = 0.6
    
    # 识别模式: nearest, voting, cluster
    recognition_mode: Literal["nearest", "voting", "cluster"] = "voting"
    
    # 投票时使用的邻居数
    voting_k: int = 5
    
    # 是否启用活体检测
    enable_liveness: bool = False
    
    # 是否启用质量检测
    enable_quality_check: bool = True
    quality_threshold: float = 0.5


@dataclass
class WebConfig:
    """Web服务配置"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    secret_key: str = "your-secret-key-here"
    
    # 上传文件配置
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: Tuple[str, ...] = ("png", "jpg", "jpeg", "gif", "bmp")
    
    # 静态文件
    static_folder: str = str(BASE_DIR / "web" / "static")
    template_folder: str = str(BASE_DIR / "web" / "templates")


@dataclass
class SystemConfig:
    """系统总配置"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = str(BASE_DIR / "logs" / "system.log")


# 全局配置实例
config = SystemConfig()


def init_directories():
    """初始化必要的目录"""
    dirs = [
        Path(config.database.face_images_dir),
        Path(config.database.original_images_dir),
        Path(config.database.db_path).parent,
        Path(config.log_file).parent,
        BASE_DIR / "data" / "uploads",
        BASE_DIR / "data" / "temp",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# 启动时初始化目录
init_directories()

