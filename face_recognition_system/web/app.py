"""
Flask Web Application
人脸识别系统Web服务

API端点:
- /api/detect - 人脸检测
- /api/recognize - 人脸识别
- /api/register - 人员注册
- /api/cluster - 批量聚类
- /api/persons - 人员管理
- /api/faces - 人脸管理
- /api/stats - 统计信息
"""
import os
import sys
import base64
import uuid
import logging
from pathlib import Path
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config, init_directories
from core.detector import FaceDetector
from core.feature_extractor import FeatureExtractor
from core.clustering_engine import DGFCClusteringEngine, DGFCConfig
from core.recognition_engine import RecognitionEngine
from database.face_db import FaceDatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
app.config["SECRET_KEY"] = config.web.secret_key
app.config["MAX_CONTENT_LENGTH"] = config.web.max_content_length
CORS(app)

# 全局变量
recognition_engine = None
db_manager = None
UPLOAD_FOLDER = Path(config.database.original_images_dir) / "uploads"


def init_app():
    """初始化应用"""
    global recognition_engine, db_manager
    
    init_directories()
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Recognition Engine...")
    
    try:
        recognition_engine = RecognitionEngine(
            detector_type=config.detection.detector_type,
            feature_model=config.feature.model_type,
            device=config.feature.device,
            similarity_threshold=config.recognition.similarity_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to initialize recognition engine: {e}")
        recognition_engine = None
    
    logger.info("Initializing Database Manager...")
    db_manager = FaceDatabaseManager(
        db_path=config.database.db_path,
        faiss_index_path=config.database.faiss_index_path,
        face_images_dir=config.database.face_images_dir,
        embedding_dim=config.feature.embedding_dim,
    )
    
    logger.info("Application initialized successfully")


def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.web.allowed_extensions


def decode_base64_image(base64_string):
    """解码Base64图像"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def encode_image_base64(image):
    """编码图像为Base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# ==================== API Routes ====================

@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "engine_ready": recognition_engine is not None,
    })


@app.route("/api/detect", methods=["POST"])
def detect_faces():
    """
    人脸检测API
    
    Request:
        - image: Base64图像或文件上传
        
    Response:
        - faces: 检测到的人脸列表
    """
    if recognition_engine is None:
        return jsonify({"error": "Recognition engine not initialized"}), 500
    
    # 获取图像
    image = None
    
    if "image" in request.files:
        file = request.files["image"]
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    elif request.is_json:
        data = request.get_json()
        if "image" in data:
            image = decode_base64_image(data["image"])
    
    if image is None:
        return jsonify({"error": "No valid image provided"}), 400
    
    # 检测人脸
    try:
        faces = recognition_engine.detector.detect(image)
        
        results = []
        for face in faces:
            face_data = face.to_dict()
            
            # 可选: 返回裁剪的人脸图像
            if face.aligned_face is not None:
                face_data["face_image"] = encode_image_base64(face.aligned_face)
            
            results.append(face_data)
        
        return jsonify({
            "success": True,
            "num_faces": len(results),
            "faces": results,
        })
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/recognize", methods=["POST"])
def recognize_faces():
    """
    人脸识别API
    
    Request:
        - image: Base64图像或文件上传
        
    Response:
        - results: 识别结果列表
    """
    if recognition_engine is None:
        return jsonify({"error": "Recognition engine not initialized"}), 500
    
    # 获取图像
    image = None
    
    if "image" in request.files:
        file = request.files["image"]
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    elif request.is_json:
        data = request.get_json()
        if "image" in data:
            image = decode_base64_image(data["image"])
    
    if image is None:
        return jsonify({"error": "No valid image provided"}), 400
    
    try:
        # 识别
        results = recognition_engine.process_image(image, recognize=True)
        
        response_data = []
        for result in results:
            response_data.append(result.to_dict())
        
        # 绘制结果图
        annotated_image = recognition_engine.draw_results(image, results)
        annotated_base64 = encode_image_base64(annotated_image)
        
        return jsonify({
            "success": True,
            "num_faces": len(results),
            "results": response_data,
            "annotated_image": annotated_base64,
        })
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/register", methods=["POST"])
def register_person():
    """
    注册新人员API
    
    Request:
        - name: 姓名
        - images: Base64图像列表或文件上传
        - metadata: 可选附加信息
        
    Response:
        - person: 注册的人员信息
    """
    if recognition_engine is None:
        return jsonify({"error": "Recognition engine not initialized"}), 500
    
    # 获取参数
    if request.is_json:
        data = request.get_json()
        name = data.get("name")
        image_data = data.get("images", [])
        metadata = data.get("metadata", {})
        
        images = [decode_base64_image(img) for img in image_data]
    else:
        name = request.form.get("name")
        metadata = {}
        images = []
        
        if "images" in request.files:
            files = request.files.getlist("images")
            for file in files:
                if file and allowed_file(file.filename):
                    image_bytes = file.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(image)
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    if not images:
        return jsonify({"error": "At least one image is required"}), 400
    
    try:
        # 注册
        person = recognition_engine.register_person(
            name=name,
            images=images,
            metadata=metadata,
        )
        
        if person:
            # 同步到数据库
            db_person = db_manager.create_person(
                name=name,
                metadata=metadata,
            )
            
            return jsonify({
                "success": True,
                "person": {
                    "id": person.person_id,
                    "name": person.name,
                    "num_faces": len(person.embeddings),
                },
            })
        else:
            return jsonify({"error": "No valid faces found in images"}), 400
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster", methods=["POST"])
def cluster_faces():
    """
    批量人脸聚类API
    
    Request:
        - images: Base64图像列表或文件上传
        - auto_register: 是否自动注册
        
    Response:
        - clustering_result: 聚类结果
    """
    if recognition_engine is None:
        return jsonify({"error": "Recognition engine not initialized"}), 500
    
    # 获取参数
    images = []
    auto_register = False
    
    if request.is_json:
        data = request.get_json()
        image_data = data.get("images", [])
        auto_register = data.get("auto_register", False)
        images = [decode_base64_image(img) for img in image_data]
    else:
        auto_register = request.form.get("auto_register", "false").lower() == "true"
        
        if "images" in request.files:
            files = request.files.getlist("images")
            for file in files:
                if file and allowed_file(file.filename):
                    image_bytes = file.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(image)
    
    if not images:
        return jsonify({"error": "At least one image is required"}), 400
    
    try:
        # 聚类
        result = recognition_engine.cluster_faces(
            images=images,
            auto_register=auto_register,
        )
        
        # 准备响应
        clusters = []
        for cluster_id, indices in result.cluster_indices.items():
            cluster_data = {
                "cluster_id": cluster_id,
                "num_faces": len(indices),
                "face_indices": indices,
            }
            
            # 可选: 返回每个簇的代表性人脸
            if "face_images" in result.extras:
                face_images = result.extras["face_images"]
                representative_idx = indices[0]
                if representative_idx < len(face_images):
                    cluster_data["representative_face"] = encode_image_base64(
                        face_images[representative_idx]
                    )
            
            clusters.append(cluster_data)
        
        return jsonify({
            "success": True,
            "num_clusters": result.n_clusters,
            "num_faces": len(result.labels),
            "num_core_points": int(np.sum(result.core_mask)),
            "clusters": clusters,
        })
    
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_faces():
    """
    人脸比对API
    
    比较两张图像中的人脸是否是同一人
    """
    if recognition_engine is None:
        return jsonify({"error": "Recognition engine not initialized"}), 500
    
    # 获取两张图像
    if request.is_json:
        data = request.get_json()
        image1 = decode_base64_image(data.get("image1", ""))
        image2 = decode_base64_image(data.get("image2", ""))
    else:
        return jsonify({"error": "JSON request required"}), 400
    
    if image1 is None or image2 is None:
        return jsonify({"error": "Two valid images required"}), 400
    
    try:
        is_same, similarity = recognition_engine.compare_faces(image1, image2)
        
        return jsonify({
            "success": True,
            "is_same_person": is_same,
            "similarity": similarity,
            "threshold": recognition_engine.similarity_threshold,
        })
    
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/persons", methods=["GET"])
def list_persons():
    """获取所有人员列表"""
    try:
        persons = db_manager.get_all_persons()
        return jsonify({
            "success": True,
            "persons": [p.to_dict() for p in persons],
        })
    except Exception as e:
        logger.error(f"Error listing persons: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/persons/<int:person_id>", methods=["GET"])
def get_person(person_id):
    """获取单个人员信息"""
    try:
        person = db_manager.get_person(person_id)
        if person:
            return jsonify({
                "success": True,
                "person": person.to_dict(),
            })
        else:
            return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        logger.error(f"Error getting person: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/persons/<int:person_id>", methods=["PUT"])
def update_person(person_id):
    """更新人员信息"""
    try:
        data = request.get_json()
        success = db_manager.update_person(
            person_id=person_id,
            name=data.get("name"),
            description=data.get("description"),
            metadata=data.get("metadata"),
        )
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        logger.error(f"Error updating person: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/persons/<int:person_id>", methods=["DELETE"])
def delete_person(person_id):
    """删除人员"""
    try:
        success = db_manager.delete_person(person_id)
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Person not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def get_statistics():
    """获取系统统计信息"""
    try:
        stats = db_manager.get_statistics()
        return jsonify({
            "success": True,
            "statistics": stats,
        })
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """通用图像上传"""
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = UPLOAD_FOLDER / unique_filename
        file.save(str(filepath))
        
        return jsonify({
            "success": True,
            "filename": unique_filename,
            "path": str(filepath),
        })
    
    return jsonify({"error": "Invalid file type"}), 400


# 静态文件服务
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ==================== WebSocket for Real-time ====================

try:
    from flask_socketio import SocketIO, emit
    socketio = SocketIO(app, cors_allowed_origins="*")
    SOCKETIO_AVAILABLE = True
    
    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected")
        emit("status", {"message": "Connected to server"})
    
    @socketio.on("video_frame")
    def handle_video_frame(data):
        """处理实时视频帧"""
        if recognition_engine is None:
            emit("error", {"message": "Engine not ready"})
            return
        
        try:
            image = decode_base64_image(data["frame"])
            results = recognition_engine.process_image(image, recognize=True)
            
            annotated = recognition_engine.draw_results(image, results)
            
            emit("recognition_result", {
                "frame": encode_image_base64(annotated),
                "faces": [r.to_dict() for r in results],
            })
        except Exception as e:
            emit("error", {"message": str(e)})

except ImportError:
    SOCKETIO_AVAILABLE = False
    socketio = None
    logger.warning("Flask-SocketIO not available, real-time features disabled")


# ==================== Main ====================

def run_server(host="0.0.0.0", port=5000, debug=True):
    """运行服务器"""
    init_app()
    
    if SOCKETIO_AVAILABLE and socketio:
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(
        host=config.web.host,
        port=config.web.port,
        debug=config.web.debug,
    )

