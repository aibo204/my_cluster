"""
数据库模型定义
Database Models using SQLAlchemy
"""
from datetime import datetime
from typing import Optional, List
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, LargeBinary, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Person(Base):
    """人员表"""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # 元数据 (JSON格式)
    metadata_json = Column(Text, default="{}")
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 是否激活
    is_active = Column(Boolean, default=True)
    
    # 关系
    faces = relationship("Face", back_populates="person", cascade="all, delete-orphan")
    
    @property
    def extra_data(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}
    
    @extra_data.setter
    def extra_data(self, value: dict):
        self.metadata_json = json.dumps(value)
    
    @property
    def num_faces(self) -> int:
        return len(self.faces) if self.faces else 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "extra_data": self.extra_data,
            "num_faces": self.num_faces,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }


class Face(Base):
    """人脸表"""
    __tablename__ = "faces"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 关联的人员
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    person = relationship("Person", back_populates="faces")
    
    # 特征向量 (二进制存储)
    embedding_blob = Column(LargeBinary, nullable=False)
    embedding_dim = Column(Integer, default=512)
    
    # 人脸图像路径
    face_image_path = Column(String(500), nullable=True)
    
    # 原始图像信息
    source_image_path = Column(String(500), nullable=True)
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    
    # 检测置信度
    detection_confidence = Column(Float, default=0.0)
    
    # 质量分数
    quality_score = Column(Float, default=0.0)
    
    # 聚类信息
    cluster_id = Column(Integer, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 在FAISS索引中的ID
    faiss_id = Column(Integer, nullable=True)
    
    def set_embedding(self, embedding):
        """设置特征向量"""
        import numpy as np
        self.embedding_blob = embedding.astype(np.float32).tobytes()
        self.embedding_dim = len(embedding)
    
    def get_embedding(self):
        """获取特征向量"""
        import numpy as np
        return np.frombuffer(self.embedding_blob, dtype=np.float32)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "person_id": self.person_id,
            "person_name": self.person.name if self.person else None,
            "face_image_path": self.face_image_path,
            "detection_confidence": self.detection_confidence,
            "quality_score": self.quality_score,
            "cluster_id": self.cluster_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
        }


class ClusterInfo(Base):
    """聚类信息表"""
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 聚类运行ID
    run_id = Column(String(50), nullable=False)
    
    # 聚类ID
    cluster_id = Column(Integer, nullable=False)
    
    # 包含的人脸数
    num_faces = Column(Integer, default=0)
    
    # 中心向量
    center_blob = Column(LargeBinary, nullable=True)
    
    # 是否已标注
    is_labeled = Column(Boolean, default=False)
    
    # 关联的人员ID (标注后)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    
    # 聚类质量指标
    intra_cluster_distance = Column(Float, nullable=True)
    
    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def set_center(self, center):
        """设置中心向量"""
        import numpy as np
        if center is not None:
            self.center_blob = center.astype(np.float32).tobytes()
    
    def get_center(self):
        """获取中心向量"""
        import numpy as np
        if self.center_blob:
            return np.frombuffer(self.center_blob, dtype=np.float32)
        return None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "cluster_id": self.cluster_id,
            "num_faces": self.num_faces,
            "is_labeled": self.is_labeled,
            "person_id": self.person_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RecognitionLog(Base):
    """识别日志表"""
    __tablename__ = "recognition_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 识别时间
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # 识别结果
    recognized_person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    confidence = Column(Float, default=0.0)
    is_known = Column(Boolean, default=False)
    
    # 源图像
    source_image_path = Column(String(500), nullable=True)
    
    # 人脸位置
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)
    
    # 设备信息
    device_id = Column(String(100), nullable=True)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "person_id": self.recognized_person_id,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "device_id": self.device_id,
        }


def init_database(db_path: str) -> sessionmaker:
    """初始化数据库"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session

