"""
人脸检测模块
Face Detection Module - MTCNN, RetinaFace, OpenCV

功能:
1. 多种检测器支持 (MTCNN, RetinaFace, OpenCV Cascade)
2. 人脸对齐 (基于5点关键点)
3. 人脸质量评估
4. 批量检测支持
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FaceBox:
    """人脸检测结果"""
    # 边界框 (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int]
    
    # 置信度
    confidence: float
    
    # 5点关键点 (left_eye, right_eye, nose, left_mouth, right_mouth)
    landmarks: Optional[np.ndarray] = None
    
    # 对齐后的人脸图像
    aligned_face: Optional[np.ndarray] = None
    
    # 人脸ID (用于追踪)
    face_id: Optional[int] = None
    
    # 质量分数
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "face_id": self.face_id,
            "quality_score": self.quality_score,
        }


class FaceAligner:
    """
    人脸对齐器
    基于5点关键点进行仿射变换对齐
    """
    
    # 标准5点参考位置 (112x112)
    REFERENCE_LANDMARKS_112 = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ], dtype=np.float32)
    
    # 标准5点参考位置 (160x160)
    REFERENCE_LANDMARKS_160 = np.array([
        [54.7063, 73.8514],
        [105.0454, 73.5734],
        [80.0360, 102.4808],
        [59.3563, 131.9507],
        [101.0427, 131.7195],
    ], dtype=np.float32)
    
    def __init__(self, output_size: int = 160):
        self.output_size = output_size
        if output_size == 112:
            self.reference = self.REFERENCE_LANDMARKS_112
        else:
            # 缩放到目标尺寸
            scale = output_size / 112.0
            self.reference = self.REFERENCE_LANDMARKS_112 * scale
    
    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        对齐人脸
        
        Args:
            image: BGR图像
            landmarks: 5点关键点 (5, 2)
            
        Returns:
            对齐后的人脸图像
        """
        # 计算仿射变换矩阵
        M = self._get_affine_transform(landmarks)
        
        # 应用变换
        aligned = cv2.warpAffine(
            image, 
            M, 
            (self.output_size, self.output_size),
            borderValue=(128, 128, 128)
        )
        
        return aligned
    
    def _get_affine_transform(self, src_pts: np.ndarray) -> np.ndarray:
        """计算仿射变换矩阵 (使用最小二乘法)"""
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = self.reference
        
        # 使用OpenCV的estimateAffinePartial2D更稳定
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            # 回退到简单的仿射变换
            M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        
        return M


class FaceQualityAssessor:
    """
    人脸质量评估器
    评估维度: 清晰度、姿态、光照、遮挡
    """
    
    def __init__(self):
        self.blur_threshold = 100.0
        self.brightness_range = (40, 220)
    
    def assess(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        评估人脸质量
        
        Returns:
            quality_scores: 各维度分数
        """
        scores = {}
        
        # 1. 清晰度 (Laplacian方差)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores["sharpness"] = min(laplacian_var / self.blur_threshold, 1.0)
        
        # 2. 亮度
        brightness = np.mean(gray)
        if brightness < self.brightness_range[0]:
            scores["brightness"] = brightness / self.brightness_range[0]
        elif brightness > self.brightness_range[1]:
            scores["brightness"] = (255 - brightness) / (255 - self.brightness_range[1])
        else:
            scores["brightness"] = 1.0
        
        # 3. 对比度
        contrast = np.std(gray)
        scores["contrast"] = min(contrast / 50.0, 1.0)
        
        # 4. 综合分数
        scores["overall"] = (
            scores["sharpness"] * 0.4 +
            scores["brightness"] * 0.3 +
            scores["contrast"] * 0.3
        )
        
        return scores


class MTCNNDetector:
    """MTCNN人脸检测器"""
    
    def __init__(
        self,
        image_size: int = 160,
        margin: int = 20,
        min_face_size: int = 20,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        device: str = "cpu",
    ):
        if not MTCNN_AVAILABLE:
            raise ImportError("facenet-pytorch is required for MTCNN detector")
        
        self.device = device
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            post_process=False,
            device=device,
        )
        self.image_size = image_size
        self.margin = margin
    
    def detect(
        self, 
        image: np.ndarray, 
        return_landmarks: bool = True
    ) -> List[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
        """
        检测人脸
        
        Args:
            image: BGR图像
            return_landmarks: 是否返回关键点
            
        Returns:
            List of (bbox, confidence, landmarks)
        """
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测
        boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=return_landmarks)
        
        if boxes is None:
            return []
        
        results = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            lm = landmarks[i] if landmarks is not None else None
            results.append((box.astype(int), float(prob), lm))
        
        return results
    
    def detect_and_align(
        self, 
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        检测并对齐人脸
        
        Returns:
            List of (aligned_face, bbox, confidence, landmarks)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用MTCNN的extract方法直接获取对齐后的人脸
        boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=True)
        
        if boxes is None:
            return []
        
        results = []
        aligner = FaceAligner(self.image_size)
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if landmarks is not None and landmarks[i] is not None:
                aligned = aligner.align(image, landmarks[i])
                results.append((aligned, box.astype(int), float(prob), landmarks[i]))
        
        return results


class OpenCVDetector:
    """OpenCV Cascade人脸检测器 (轻量级备选)"""
    
    def __init__(self):
        # 加载Haar级联分类器
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        # 眼睛检测器 (用于估计关键点)
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def detect(
        self, 
        image: np.ndarray, 
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
        """检测人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        
        results = []
        for (x, y, w, h) in faces:
            bbox = np.array([x, y, x + w, y + h])
            # OpenCV没有置信度,使用固定值
            confidence = 0.99
            
            # 尝试检测眼睛作为简易关键点
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            landmarks = None
            if len(eyes) >= 2:
                # 简易关键点估计
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                left_eye = (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
                right_eye = (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
                nose = (x + w//2, y + int(h*0.6))
                left_mouth = (x + int(w*0.35), y + int(h*0.85))
                right_mouth = (x + int(w*0.65), y + int(h*0.85))
                landmarks = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
            
            results.append((bbox, confidence, landmarks))
        
        return results


class FaceDetector:
    """
    统一人脸检测接口
    支持多种检测器后端
    """
    
    def __init__(
        self,
        detector_type: str = "mtcnn",
        image_size: int = 160,
        margin: int = 20,
        min_face_size: int = 20,
        confidence_threshold: float = 0.9,
        device: str = "cpu",
    ):
        """
        初始化检测器
        
        Args:
            detector_type: 检测器类型 ("mtcnn", "opencv")
            image_size: 输出人脸尺寸
            margin: 边界扩展
            min_face_size: 最小人脸尺寸
            confidence_threshold: 置信度阈值
            device: 运行设备
        """
        self.detector_type = detector_type
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        
        # 初始化检测器
        if detector_type == "mtcnn":
            self.detector = MTCNNDetector(
                image_size=image_size,
                margin=margin,
                min_face_size=min_face_size,
                device=device,
            )
        elif detector_type == "opencv":
            self.detector = OpenCVDetector()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        # 初始化对齐器和质量评估器
        self.aligner = FaceAligner(output_size=image_size)
        self.quality_assessor = FaceQualityAssessor()
    
    def detect(
        self, 
        image: Union[np.ndarray, str, Path],
        max_faces: int = 50,
        return_aligned: bool = True,
        assess_quality: bool = True,
    ) -> List[FaceBox]:
        """
        检测图像中的人脸
        
        Args:
            image: 图像(BGR数组或路径)
            max_faces: 最大人脸数
            return_aligned: 是否返回对齐后的人脸
            assess_quality: 是否评估质量
            
        Returns:
            检测到的人脸列表
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                logger.error(f"Failed to load image: {image}")
                return []
        
        # 检测
        detections = self.detector.detect(image, return_landmarks=True)
        
        # 过滤和处理
        faces = []
        for i, (bbox, conf, landmarks) in enumerate(detections):
            if conf < self.confidence_threshold:
                continue
            
            if len(faces) >= max_faces:
                break
            
            face_box = FaceBox(
                bbox=tuple(bbox),
                confidence=conf,
                landmarks=landmarks,
                face_id=i,
            )
            
            # 对齐
            if return_aligned and landmarks is not None:
                try:
                    aligned = self.aligner.align(image, landmarks)
                    face_box.aligned_face = aligned
                except Exception as e:
                    logger.warning(f"Failed to align face: {e}")
            
            # 质量评估
            if assess_quality and face_box.aligned_face is not None:
                try:
                    quality = self.quality_assessor.assess(face_box.aligned_face)
                    face_box.quality_score = quality["overall"]
                except Exception as e:
                    logger.warning(f"Failed to assess quality: {e}")
            
            faces.append(face_box)
        
        return faces
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        **kwargs
    ) -> List[List[FaceBox]]:
        """批量检测"""
        return [self.detect(img, **kwargs) for img in images]
    
    def detect_from_video_frame(
        self,
        frame: np.ndarray,
        **kwargs
    ) -> List[FaceBox]:
        """从视频帧检测 (与detect相同,但可扩展追踪功能)"""
        return self.detect(frame, **kwargs)

