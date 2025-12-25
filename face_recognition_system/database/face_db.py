"""
人脸数据库管理器
Face Database Manager with FAISS Vector Index
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import pickle
import uuid

from sqlalchemy.orm import Session

from .models import Person, Face, ClusterInfo, RecognitionLog, init_database

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using brute-force search")


class FAISSIndex:
    """
    FAISS向量索引
    支持快速最近邻搜索
    """
    
    def __init__(
        self,
        dim: int = 512,
        index_type: str = "IVF",  # "Flat", "IVF", "HNSW"
        nlist: int = 100,
        use_gpu: bool = False,
    ):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        self._index = None
        self._id_map: Dict[int, int] = {}  # faiss_id -> face_id
        self._next_id = 0
        
        self._build_index()
    
    def _build_index(self):
        """构建索引"""
        if not FAISS_AVAILABLE:
            self._index = None
            return
        
        if self.index_type == "Flat":
            # 精确搜索
            self._index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "IVF":
            # IVF索引 (需要训练)
            quantizer = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self._index.nprobe = 10
        elif self.index_type == "HNSW":
            # HNSW图索引
            self._index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            self._index = faiss.IndexFlatIP(self.dim)
        
        # 包装为ID映射索引
        if self.index_type != "Flat":
            self._index = faiss.IndexIDMap(self._index)
        
        logger.info(f"FAISS index created: {self.index_type}")
    
    def add(self, embedding: np.ndarray, face_id: int) -> int:
        """添加向量"""
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        if FAISS_AVAILABLE and self._index is not None:
            faiss_id = self._next_id
            self._next_id += 1
            
            if hasattr(self._index, "add_with_ids"):
                self._index.add_with_ids(embedding, np.array([faiss_id], dtype=np.int64))
            else:
                self._index.add(embedding)
            
            self._id_map[faiss_id] = face_id
            return faiss_id
        else:
            # 简单列表存储
            if not hasattr(self, "_vectors"):
                self._vectors = []
                self._face_ids = []
            self._vectors.append(embedding.flatten())
            self._face_ids.append(face_id)
            return len(self._vectors) - 1
    
    def add_batch(self, embeddings: np.ndarray, face_ids: List[int]):
        """批量添加"""
        for emb, fid in zip(embeddings, face_ids):
            self.add(emb, fid)
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """搜索最相似的向量"""
        query = query.astype(np.float32).reshape(1, -1)
        
        if FAISS_AVAILABLE and self._index is not None:
            if self._index.ntotal == 0:
                return []
            
            k = min(k, self._index.ntotal)
            distances, indices = self._index.search(query, k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx in self._id_map:
                    results.append((self._id_map[idx], float(dist)))
            
            return results
        else:
            # 暴力搜索
            if not hasattr(self, "_vectors") or not self._vectors:
                return []
            
            vectors = np.array(self._vectors)
            similarities = np.dot(vectors, query.flatten())
            
            indices = np.argsort(similarities)[::-1][:k]
            return [(self._face_ids[i], float(similarities[i])) for i in indices]
    
    def remove(self, faiss_id: int):
        """删除向量 (FAISS的删除效率较低)"""
        if faiss_id in self._id_map:
            del self._id_map[faiss_id]
    
    def save(self, path: str):
        """保存索引"""
        if FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, f"{path}.index")
        
        with open(f"{path}.map", "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "next_id": self._next_id,
            }, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str):
        """加载索引"""
        if FAISS_AVAILABLE and Path(f"{path}.index").exists():
            self._index = faiss.read_index(f"{path}.index")
        
        map_path = Path(f"{path}.map")
        if map_path.exists():
            with open(map_path, "rb") as f:
                data = pickle.load(f)
                self._id_map = data["id_map"]
                self._next_id = data["next_id"]
        
        logger.info(f"Index loaded from {path}")
    
    @property
    def size(self) -> int:
        if FAISS_AVAILABLE and self._index is not None:
            return self._index.ntotal
        elif hasattr(self, "_vectors"):
            return len(self._vectors)
        return 0


class FaceDatabaseManager:
    """
    人脸数据库管理器
    
    功能:
    1. 人员CRUD操作
    2. 人脸CRUD操作
    3. 向量索引管理
    4. 聚类结果管理
    5. 识别日志记录
    """
    
    def __init__(
        self,
        db_path: str,
        faiss_index_path: str,
        face_images_dir: str,
        embedding_dim: int = 512,
    ):
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.face_images_dir = Path(face_images_dir)
        self.embedding_dim = embedding_dim
        
        # 确保目录存在
        self.face_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self.SessionLocal = init_database(db_path)
        
        # 初始化向量索引
        self.vector_index = FAISSIndex(dim=embedding_dim)
        
        # 尝试加载已有索引
        if Path(f"{faiss_index_path}.map").exists():
            self.vector_index.load(faiss_index_path)
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    # ============ Person Operations ============
    
    def create_person(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Person:
        """创建人员"""
        with self.get_session() as session:
            person = Person(
                name=name,
                description=description,
            )
            if metadata:
                person.extra_data = metadata
            
            session.add(person)
            session.commit()
            session.refresh(person)
            
            logger.info(f"Created person: {person.id} - {name}")
            return person
    
    def get_person(self, person_id: int) -> Optional[Person]:
        """获取人员"""
        with self.get_session() as session:
            return session.query(Person).filter(Person.id == person_id).first()
    
    def get_all_persons(self, active_only: bool = True) -> List[Person]:
        """获取所有人员"""
        with self.get_session() as session:
            query = session.query(Person)
            if active_only:
                query = query.filter(Person.is_active == True)
            return query.all()
    
    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """更新人员信息"""
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if not person:
                return False
            
            if name:
                person.name = name
            if description:
                person.description = description
            if metadata:
                person.extra_data = metadata
            
            session.commit()
            return True
    
    def delete_person(self, person_id: int) -> bool:
        """删除人员 (软删除)"""
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if not person:
                return False
            
            person.is_active = False
            session.commit()
            return True
    
    # ============ Face Operations ============
    
    def add_face(
        self,
        embedding: np.ndarray,
        person_id: Optional[int] = None,
        face_image: Optional[np.ndarray] = None,
        source_image_path: Optional[str] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        detection_confidence: float = 0.0,
        quality_score: float = 0.0,
    ) -> Face:
        """添加人脸"""
        import cv2
        
        with self.get_session() as session:
            face = Face(
                person_id=person_id,
                source_image_path=source_image_path,
                detection_confidence=detection_confidence,
                quality_score=quality_score,
            )
            face.set_embedding(embedding)
            
            if bbox:
                face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2 = bbox
            
            # 保存人脸图像
            if face_image is not None:
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = self.face_images_dir / filename
                cv2.imwrite(str(filepath), face_image)
                face.face_image_path = str(filepath)
            
            session.add(face)
            session.commit()
            session.refresh(face)
            
            # 添加到向量索引
            faiss_id = self.vector_index.add(embedding, face.id)
            face.faiss_id = faiss_id
            session.commit()
            
            return face
    
    def get_face(self, face_id: int) -> Optional[Face]:
        """获取人脸"""
        with self.get_session() as session:
            return session.query(Face).filter(Face.id == face_id).first()
    
    def get_faces_by_person(self, person_id: int) -> List[Face]:
        """获取人员的所有人脸"""
        with self.get_session() as session:
            return session.query(Face).filter(Face.person_id == person_id).all()
    
    def assign_face_to_person(self, face_id: int, person_id: int) -> bool:
        """将人脸分配给人员"""
        with self.get_session() as session:
            face = session.query(Face).filter(Face.id == face_id).first()
            if not face:
                return False
            
            face.person_id = person_id
            session.commit()
            return True
    
    def search_faces(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[Face, float]]:
        """搜索相似人脸"""
        results = self.vector_index.search(query_embedding, k=top_k)
        
        faces = []
        with self.get_session() as session:
            for face_id, score in results:
                if score >= threshold:
                    face = session.query(Face).filter(Face.id == face_id).first()
                    if face:
                        faces.append((face, score))
        
        return faces
    
    # ============ Clustering Operations ============
    
    def save_clustering_result(
        self,
        labels: np.ndarray,
        face_ids: List[int],
        cluster_centers: Optional[np.ndarray] = None,
    ) -> str:
        """保存聚类结果"""
        run_id = uuid.uuid4().hex[:8]
        
        with self.get_session() as session:
            # 更新人脸的聚类标签
            for face_id, label in zip(face_ids, labels):
                face = session.query(Face).filter(Face.id == face_id).first()
                if face:
                    face.cluster_id = int(label)
            
            # 保存聚类信息
            unique_labels = np.unique(labels[labels > 0])
            for i, label in enumerate(unique_labels):
                mask = labels == label
                num_faces = int(np.sum(mask))
                
                cluster_info = ClusterInfo(
                    run_id=run_id,
                    cluster_id=int(label),
                    num_faces=num_faces,
                )
                
                if cluster_centers is not None and i < len(cluster_centers):
                    cluster_info.set_center(cluster_centers[i])
                
                session.add(cluster_info)
            
            session.commit()
        
        logger.info(f"Saved clustering result: run_id={run_id}")
        return run_id
    
    def get_clustering_result(self, run_id: str) -> List[ClusterInfo]:
        """获取聚类结果"""
        with self.get_session() as session:
            return session.query(ClusterInfo).filter(
                ClusterInfo.run_id == run_id
            ).all()
    
    def label_cluster(
        self,
        cluster_id: int,
        run_id: str,
        person_id: int,
    ) -> bool:
        """标注聚类 (将聚类关联到人员)"""
        with self.get_session() as session:
            # 更新聚类信息
            cluster = session.query(ClusterInfo).filter(
                ClusterInfo.cluster_id == cluster_id,
                ClusterInfo.run_id == run_id,
            ).first()
            
            if cluster:
                cluster.is_labeled = True
                cluster.person_id = person_id
            
            # 更新所有属于该聚类的人脸
            faces = session.query(Face).filter(
                Face.cluster_id == cluster_id
            ).all()
            
            for face in faces:
                face.person_id = person_id
            
            session.commit()
            return True
    
    # ============ Logging ============
    
    def log_recognition(
        self,
        person_id: Optional[int],
        confidence: float,
        is_known: bool,
        source_image_path: Optional[str] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        device_id: Optional[str] = None,
    ):
        """记录识别日志"""
        with self.get_session() as session:
            log = RecognitionLog(
                recognized_person_id=person_id,
                confidence=confidence,
                is_known=is_known,
                source_image_path=source_image_path,
                device_id=device_id,
            )
            
            if bbox:
                log.bbox_x1, log.bbox_y1, log.bbox_x2, log.bbox_y2 = bbox
            
            session.add(log)
            session.commit()
    
    # ============ Statistics ============
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.get_session() as session:
            num_persons = session.query(Person).filter(Person.is_active == True).count()
            num_faces = session.query(Face).count()
            num_unassigned = session.query(Face).filter(Face.person_id == None).count()
            
            return {
                "num_persons": num_persons,
                "num_faces": num_faces,
                "num_unassigned_faces": num_unassigned,
                "index_size": self.vector_index.size,
            }
    
    def save_index(self):
        """保存向量索引"""
        self.vector_index.save(self.faiss_index_path)
    
    def close(self):
        """关闭数据库连接"""
        self.save_index()

