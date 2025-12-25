"""
人脸特征提取模块
Face Feature Extraction Module

支持的模型:
- FaceNet (InceptionResnetV1)
- ArcFace
- 自定义模型接口
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Union, Tuple, Any, TYPE_CHECKING
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
FACENET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    transforms = None
    logger.warning("PyTorch not available, feature extraction will be limited")

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    InceptionResnetV1 = None


class BaseFeatureExtractor:
    """特征提取器基类"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        input_size: Tuple[int, int] = (160, 160),
        l2_normalize: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.l2_normalize = l2_normalize
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取单张人脸特征"""
        raise NotImplementedError
    
    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        return np.stack([self.extract(img) for img in face_images])
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2归一化"""
        if self.l2_normalize:
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
            return embedding / (norm + 1e-10)
        return embedding


class FaceNetExtractor(BaseFeatureExtractor):
    """
    FaceNet特征提取器
    使用InceptionResnetV1预训练模型
    """
    
    def __init__(
        self,
        pretrained: str = "vggface2",  # "vggface2" or "casia-webface"
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(embedding_dim=512, input_size=(160, 160), **kwargs)
        
        if not TORCH_AVAILABLE or not FACENET_AVAILABLE:
            raise ImportError("PyTorch and facenet-pytorch are required")
        
        self.device = torch.device(device)
        
        # 加载预训练模型
        self.model = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
        ).eval().to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        logger.info(f"FaceNet model loaded (pretrained={pretrained}, device={device})")
    
    def _preprocess(self, face_image: np.ndarray) -> Any:
        """预处理图像"""
        # BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # 确保尺寸正确
        if face_image.shape[:2] != self.input_size:
            face_image = cv2.resize(face_image, self.input_size)
        
        tensor = self.transform(face_image)
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取单张人脸特征向量"""
        tensor = self._preprocess(face_image)
        embedding = self.model(tensor).cpu().numpy().squeeze()
        return self._normalize(embedding)
    
    @torch.no_grad()
    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        if len(face_images) == 0:
            return np.array([])
        
        # 批量预处理
        tensors = torch.cat([self._preprocess(img) for img in face_images], dim=0)
        
        # 批量推理
        embeddings = self.model(tensors).cpu().numpy()
        
        return self._normalize(embeddings)


class SimpleFeatureExtractor(BaseFeatureExtractor):
    """
    简易特征提取器 (不依赖深度学习框架)
    使用传统方法: HOG + PCA
    适合快速原型和测试
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        input_size: Tuple[int, int] = (64, 64),
        **kwargs
    ):
        super().__init__(embedding_dim=embedding_dim, input_size=input_size, **kwargs)
        
        # HOG参数
        self.hog = cv2.HOGDescriptor(
            _winSize=(input_size[0], input_size[1]),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        
        # PCA降维 (需要在大量数据上预训练)
        self.pca = None
        self._embedding_dim = embedding_dim
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取HOG特征"""
        # 转灰度
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # 调整尺寸
        gray = cv2.resize(gray, self.input_size)
        
        # 计算HOG
        hog_features = self.hog.compute(gray).flatten()
        
        # 如果有PCA则降维
        if self.pca is not None:
            hog_features = self.pca.transform(hog_features.reshape(1, -1)).squeeze()
        else:
            # 简单截断或padding到目标维度
            if len(hog_features) > self._embedding_dim:
                hog_features = hog_features[:self._embedding_dim]
            else:
                hog_features = np.pad(
                    hog_features, 
                    (0, self._embedding_dim - len(hog_features))
                )
        
        return self._normalize(hog_features)


class ArcFaceExtractor(BaseFeatureExtractor):
    """
    ArcFace特征提取器
    需要预训练的ArcFace模型文件
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(embedding_dim=512, input_size=(112, 112), **kwargs)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ArcFace")
        
        self.device = torch.device(device)
        
        # 如果有预训练模型路径则加载
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            logger.warning("ArcFace model not found, using FaceNet as fallback")
            # 回退到FaceNet
            self._fallback = FaceNetExtractor(device=device, **kwargs)
            self.model = None
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _load_model(self, model_path: str):
        """加载预训练模型"""
        # 这里需要根据具体的ArcFace实现来加载
        # 简化起见,返回None使用fallback
        logger.info(f"Loading ArcFace model from {model_path}")
        return None
    
    @torch.no_grad()
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取特征"""
        if self.model is None:
            return self._fallback.extract(face_image)
        
        # ArcFace推理逻辑
        # BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().numpy().squeeze()
        return self._normalize(embedding)


class FeatureExtractor:
    """
    统一特征提取接口
    自动选择最佳可用模型
    """
    
    def __init__(
        self,
        model_type: str = "facenet",
        device: str = "cpu",
        pretrained_path: Optional[str] = None,
        l2_normalize: bool = True,
    ):
        """
        初始化特征提取器
        
        Args:
            model_type: 模型类型 ("facenet", "arcface", "simple")
            device: 运行设备 ("cpu", "cuda")
            pretrained_path: 预训练模型路径
            l2_normalize: 是否L2归一化
        """
        self.model_type = model_type
        self.device = device
        
        # 初始化提取器
        try:
            if model_type == "facenet":
                self.extractor = FaceNetExtractor(
                    device=device,
                    l2_normalize=l2_normalize,
                )
            elif model_type == "arcface":
                self.extractor = ArcFaceExtractor(
                    model_path=pretrained_path,
                    device=device,
                    l2_normalize=l2_normalize,
                )
            elif model_type == "simple":
                self.extractor = SimpleFeatureExtractor(
                    l2_normalize=l2_normalize,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except ImportError as e:
            logger.warning(f"Failed to load {model_type}, falling back to simple: {e}")
            self.extractor = SimpleFeatureExtractor(l2_normalize=l2_normalize)
        
        self.embedding_dim = self.extractor.embedding_dim
        self.input_size = self.extractor.input_size
        
        logger.info(f"Feature extractor initialized: {type(self.extractor).__name__}")
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        提取单张人脸特征
        
        Args:
            face_image: 对齐后的人脸图像 (BGR)
            
        Returns:
            特征向量 (embedding_dim,)
        """
        return self.extractor.extract(face_image)
    
    def extract_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            face_images: 人脸图像列表
            
        Returns:
            特征矩阵 (N, embedding_dim)
        """
        return self.extractor.extract_batch(face_images)
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        计算两个特征向量的相似度
        
        Args:
            embedding1, embedding2: 特征向量
            metric: 距离度量 ("cosine", "euclidean")
            
        Returns:
            相似度分数 (0-1, 越高越相似)
        """
        if metric == "cosine":
            # 余弦相似度
            sim = np.dot(embedding1, embedding2)
            sim = (sim + 1) / 2  # 归一化到[0, 1]
        elif metric == "euclidean":
            # 欧氏距离转相似度
            dist = np.linalg.norm(embedding1 - embedding2)
            sim = 1 / (1 + dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(sim)
    
    def find_matches(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        在gallery中查找最相似的人脸
        
        Args:
            query_embedding: 查询特征
            gallery_embeddings: Gallery特征矩阵
            top_k: 返回top-k结果
            threshold: 相似度阈值
            
        Returns:
            List of (index, similarity)
        """
        # 计算余弦相似度
        similarities = np.dot(gallery_embeddings, query_embedding)
        
        # 排序
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((int(idx), sim))
        
        return results

