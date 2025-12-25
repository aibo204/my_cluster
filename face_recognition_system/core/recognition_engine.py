"""
人脸识别引擎
Face Recognition Engine

集成检测、特征提取、聚类和识别的完整流程
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import cv2
import time

from .detector import FaceDetector, FaceBox
from .feature_extractor import FeatureExtractor
from .clustering_engine import DGFCClusteringEngine, DGFCConfig, ClusteringResult

logger = logging.getLogger(__name__)


@dataclass
class Person:
    """人员信息"""
    person_id: int
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    face_images: List[np.ndarray] = field(default_factory=list)
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def representative_embedding(self) -> Optional[np.ndarray]:
        """代表性特征 (所有特征的平均)"""
        if not self.embeddings:
            return None
        mean = np.mean(self.embeddings, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-10)


@dataclass
class RecognitionResult:
    """识别结果"""
    face_box: FaceBox
    person: Optional[Person]
    confidence: float
    is_known: bool
    alternatives: List[Tuple[Person, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.face_box.bbox),
            "face_confidence": self.face_box.confidence,
            "person_id": self.person.person_id if self.person else None,
            "person_name": self.person.name if self.person else "Unknown",
            "recognition_confidence": self.confidence,
            "is_known": self.is_known,
        }


class FaceDatabase:
    """
    人脸数据库
    管理已注册的人员和特征
    """
    
    def __init__(self):
        self.persons: Dict[int, Person] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._embedding_to_person: Dict[int, int] = {}  # embedding_idx -> person_id
        self._next_person_id = 1
        self._dirty = True  # 是否需要重建索引
    
    def add_person(
        self, 
        name: str,
        embeddings: Optional[List[np.ndarray]] = None,
        face_images: Optional[List[np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Person:
        """添加新人员"""
        person = Person(
            person_id=self._next_person_id,
            name=name,
            embeddings=embeddings or [],
            face_images=face_images or [],
            metadata=metadata or {},
        )
        self.persons[person.person_id] = person
        self._next_person_id += 1
        self._dirty = True
        return person
    
    def add_face_to_person(
        self,
        person_id: int,
        embedding: np.ndarray,
        face_image: Optional[np.ndarray] = None,
    ) -> bool:
        """为已有人员添加人脸"""
        if person_id not in self.persons:
            return False
        
        self.persons[person_id].embeddings.append(embedding)
        if face_image is not None:
            self.persons[person_id].face_images.append(face_image)
        
        self._dirty = True
        return True
    
    def remove_person(self, person_id: int) -> bool:
        """删除人员"""
        if person_id in self.persons:
            del self.persons[person_id]
            self._dirty = True
            return True
        return False
    
    def get_person(self, person_id: int) -> Optional[Person]:
        """获取人员信息"""
        return self.persons.get(person_id)
    
    def get_all_persons(self) -> List[Person]:
        """获取所有人员"""
        return list(self.persons.values())
    
    def _rebuild_index(self):
        """重建特征索引"""
        embeddings = []
        self._embedding_to_person.clear()
        
        idx = 0
        for person in self.persons.values():
            for emb in person.embeddings:
                embeddings.append(emb)
                self._embedding_to_person[idx] = person.person_id
                idx += 1
        
        if embeddings:
            self._embeddings_matrix = np.array(embeddings)
        else:
            self._embeddings_matrix = None
        
        self._dirty = False
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[Person, float]]:
        """搜索最相似的人员"""
        if self._dirty:
            self._rebuild_index()
        
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            return []
        
        # 余弦相似度
        similarities = np.dot(self._embeddings_matrix, query_embedding)
        
        # 聚合到人员级别 (取每个人的最大相似度)
        person_scores: Dict[int, float] = {}
        for idx, sim in enumerate(similarities):
            pid = self._embedding_to_person[idx]
            if pid not in person_scores or sim > person_scores[pid]:
                person_scores[pid] = float(sim)
        
        # 排序
        sorted_persons = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for pid, score in sorted_persons[:top_k]:
            if score >= threshold:
                results.append((self.persons[pid], score))
        
        return results
    
    @property
    def num_persons(self) -> int:
        return len(self.persons)
    
    @property
    def num_faces(self) -> int:
        return sum(len(p.embeddings) for p in self.persons.values())


class RecognitionEngine:
    """
    人脸识别引擎
    
    功能:
    1. 人脸检测与对齐
    2. 特征提取
    3. 人脸识别 (1:N搜索)
    4. 人脸聚类
    5. 人员注册与管理
    """
    
    def __init__(
        self,
        detector_type: str = "mtcnn",
        feature_model: str = "facenet",
        device: str = "cpu",
        similarity_threshold: float = 0.6,
        clustering_config: Optional[DGFCConfig] = None,
    ):
        """
        初始化识别引擎
        
        Args:
            detector_type: 检测器类型
            feature_model: 特征提取模型
            device: 运行设备
            similarity_threshold: 识别相似度阈值
            clustering_config: 聚类配置
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # 初始化组件
        logger.info("Initializing Face Detector...")
        self.detector = FaceDetector(
            detector_type=detector_type,
            device=device,
        )
        
        logger.info("Initializing Feature Extractor...")
        self.extractor = FeatureExtractor(
            model_type=feature_model,
            device=device,
        )
        
        logger.info("Initializing Clustering Engine...")
        self.clustering_engine = DGFCClusteringEngine(
            config=clustering_config or DGFCConfig()
        )
        
        # 人脸数据库
        self.database = FaceDatabase()
        
        logger.info("Recognition Engine initialized successfully")
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        recognize: bool = True,
    ) -> List[RecognitionResult]:
        """
        处理单张图像
        
        Args:
            image: 输入图像
            recognize: 是否执行识别
            
        Returns:
            识别结果列表
        """
        # 1. 检测人脸
        faces = self.detector.detect(image, return_aligned=True)
        
        results = []
        for face in faces:
            if face.aligned_face is None:
                continue
            
            # 2. 提取特征
            embedding = self.extractor.extract(face.aligned_face)
            
            if recognize and self.database.num_persons > 0:
                # 3. 识别
                matches = self.database.search(
                    embedding, 
                    top_k=5, 
                    threshold=self.similarity_threshold
                )
                
                if matches:
                    best_person, best_score = matches[0]
                    result = RecognitionResult(
                        face_box=face,
                        person=best_person,
                        confidence=best_score,
                        is_known=True,
                        alternatives=matches[1:],
                    )
                else:
                    result = RecognitionResult(
                        face_box=face,
                        person=None,
                        confidence=0.0,
                        is_known=False,
                    )
            else:
                result = RecognitionResult(
                    face_box=face,
                    person=None,
                    confidence=0.0,
                    is_known=False,
                )
            
            results.append(result)
        
        return results
    
    def register_person(
        self,
        name: str,
        images: List[Union[np.ndarray, str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Person]:
        """
        注册新人员
        
        Args:
            name: 姓名
            images: 人脸图像列表
            metadata: 附加信息
            
        Returns:
            注册的人员对象
        """
        embeddings = []
        face_images = []
        
        for img in images:
            faces = self.detector.detect(img, return_aligned=True)
            if not faces:
                continue
            
            # 取最大的人脸
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            if face.aligned_face is not None:
                embedding = self.extractor.extract(face.aligned_face)
                embeddings.append(embedding)
                face_images.append(face.aligned_face)
        
        if not embeddings:
            logger.warning(f"No valid faces found for person: {name}")
            return None
        
        person = self.database.add_person(
            name=name,
            embeddings=embeddings,
            face_images=face_images,
            metadata=metadata or {},
        )
        
        logger.info(f"Registered person: {name} with {len(embeddings)} faces")
        return person
    
    def cluster_faces(
        self,
        images: List[Union[np.ndarray, str, Path]],
        auto_register: bool = False,
    ) -> ClusteringResult:
        """
        对多张图片中的人脸进行聚类
        
        Args:
            images: 图像列表
            auto_register: 是否自动将聚类结果注册为人员
            
        Returns:
            聚类结果
        """
        all_embeddings = []
        all_face_images = []
        all_face_boxes = []
        image_indices = []
        
        logger.info(f"Extracting faces from {len(images)} images...")
        
        for img_idx, img in enumerate(images):
            faces = self.detector.detect(img, return_aligned=True)
            
            for face in faces:
                if face.aligned_face is None:
                    continue
                
                embedding = self.extractor.extract(face.aligned_face)
                all_embeddings.append(embedding)
                all_face_images.append(face.aligned_face)
                all_face_boxes.append(face)
                image_indices.append(img_idx)
        
        if not all_embeddings:
            logger.warning("No faces found in images")
            return ClusteringResult(
                labels=np.array([]),
                n_clusters=0,
                core_mask=np.array([]),
            )
        
        embeddings_matrix = np.array(all_embeddings)
        
        logger.info(f"Clustering {len(all_embeddings)} faces...")
        result = self.clustering_engine.fit_predict(embeddings_matrix)
        
        # 添加额外信息
        result.extras["face_images"] = all_face_images
        result.extras["face_boxes"] = all_face_boxes
        result.extras["image_indices"] = image_indices
        
        if auto_register:
            self._auto_register_clusters(result, all_embeddings, all_face_images)
        
        return result
    
    def _auto_register_clusters(
        self,
        result: ClusteringResult,
        embeddings: List[np.ndarray],
        face_images: List[np.ndarray],
    ):
        """将聚类结果自动注册为人员"""
        for cluster_id, indices in result.cluster_indices.items():
            if len(indices) < 2:  # 至少需要2张脸
                continue
            
            cluster_embeddings = [embeddings[i] for i in indices]
            cluster_faces = [face_images[i] for i in indices]
            
            # 使用聚类ID作为临时名称
            name = f"Person_{cluster_id}"
            
            self.database.add_person(
                name=name,
                embeddings=cluster_embeddings,
                face_images=cluster_faces,
                metadata={"auto_registered": True, "cluster_id": cluster_id},
            )
        
        logger.info(f"Auto-registered {result.n_clusters} persons from clustering")
    
    def compare_faces(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
    ) -> Tuple[bool, float]:
        """
        比较两张图像中的人脸是否是同一人
        
        Returns:
            (is_same, similarity)
        """
        faces1 = self.detector.detect(image1, return_aligned=True, max_faces=1)
        faces2 = self.detector.detect(image2, return_aligned=True, max_faces=1)
        
        if not faces1 or not faces2:
            return False, 0.0
        
        if faces1[0].aligned_face is None or faces2[0].aligned_face is None:
            return False, 0.0
        
        emb1 = self.extractor.extract(faces1[0].aligned_face)
        emb2 = self.extractor.extract(faces2[0].aligned_face)
        
        similarity = float(np.dot(emb1, emb2))
        is_same = similarity >= self.similarity_threshold
        
        return is_same, similarity
    
    def draw_results(
        self,
        image: np.ndarray,
        results: List[RecognitionResult],
        draw_landmarks: bool = False,
    ) -> np.ndarray:
        """
        在图像上绘制识别结果
        """
        img = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result.face_box.bbox
            
            # 颜色: 绿色=已知, 红色=未知
            color = (0, 255, 0) if result.is_known else (0, 0, 255)
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if result.is_known and result.person:
                label = f"{result.person.name} ({result.confidence:.2f})"
            else:
                label = f"Unknown ({result.face_box.confidence:.2f})"
            
            # 标签背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 绘制关键点
            if draw_landmarks and result.face_box.landmarks is not None:
                for (lx, ly) in result.face_box.landmarks.astype(int):
                    cv2.circle(img, (lx, ly), 2, (255, 0, 0), -1)
        
        return img
    
    def save_database(self, path: Union[str, Path]):
        """保存数据库"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.database, f)
        logger.info(f"Database saved to {path}")
    
    def load_database(self, path: Union[str, Path]):
        """加载数据库"""
        import pickle
        with open(path, "rb") as f:
            self.database = pickle.load(f)
        logger.info(f"Database loaded from {path} ({self.database.num_persons} persons)")


class RealtimeRecognizer:
    """
    实时人脸识别器
    优化用于视频流处理
    """
    
    def __init__(
        self,
        engine: RecognitionEngine,
        skip_frames: int = 2,
        track_faces: bool = True,
    ):
        self.engine = engine
        self.skip_frames = skip_frames
        self.track_faces = track_faces
        self.frame_count = 0
        self._last_results: List[RecognitionResult] = []
    
    def process_frame(self, frame: np.ndarray) -> List[RecognitionResult]:
        """处理视频帧"""
        self.frame_count += 1
        
        # 跳帧策略
        if self.frame_count % (self.skip_frames + 1) != 0:
            return self._last_results
        
        # 处理
        results = self.engine.process_image(frame, recognize=True)
        self._last_results = results
        
        return results
    
    def run_camera(
        self,
        camera_id: int = 0,
        window_name: str = "Face Recognition",
    ):
        """运行摄像头实时识别"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        fps_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理
                results = self.process_frame(frame)
                
                # 绘制结果
                display = self.engine.draw_results(frame, results)
                
                # 计算FPS
                if time.time() - fps_time >= 1.0:
                    fps = self.frame_count
                    self.frame_count = 0
                    fps_time = time.time()
                
                # 显示FPS
                cv2.putText(
                    display, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                cv2.imshow(window_name, display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

