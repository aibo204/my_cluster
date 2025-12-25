"""
核心模块
Face Recognition System Core Modules
"""
# 延迟导入以避免依赖问题
__all__ = [
    "FaceDetector",
    "FeatureExtractor", 
    "DGFCClusteringEngine",
    "RecognitionEngine",
]

def __getattr__(name):
    """延迟导入"""
    if name == "FaceDetector":
        from .detector import FaceDetector
        return FaceDetector
    elif name == "FeatureExtractor":
        from .feature_extractor import FeatureExtractor
        return FeatureExtractor
    elif name == "DGFCClusteringEngine":
        from .clustering_engine import DGFCClusteringEngine
        return DGFCClusteringEngine
    elif name == "RecognitionEngine":
        from .recognition_engine import RecognitionEngine
        return RecognitionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

