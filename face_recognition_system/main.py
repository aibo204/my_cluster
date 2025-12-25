#!/usr/bin/env python3
"""
DGFC人脸识别系统 - 主入口
DGFC Face Recognition System - Main Entry Point

基于密度聚类的智能人脸识别系统

Usage:
    python main.py                  # 启动Web服务
    python main.py --demo           # 运行演示
    python main.py --cli            # 命令行模式
    python main.py --camera         # 摄像头实时识别
"""
import argparse
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import config, init_directories

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.log_file, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def run_web_server():
    """运行Web服务器"""
    from web.app import run_server
    
    logger.info("Starting DGFC Face Recognition Web Server...")
    logger.info(f"Server URL: http://{config.web.host}:{config.web.port}")
    
    run_server(
        host=config.web.host,
        port=config.web.port,
        debug=config.web.debug,
    )


def run_demo():
    """运行演示程序"""
    import numpy as np
    
    print("\n" + "=" * 60)
    print("       DGFC人脸识别系统演示 - 聚类算法测试")
    print("=" * 60)
    
    # 演示DGFC聚类算法
    print("\n[1] 测试DGFC聚类算法...")
    print("    生成模拟人脸特征数据 (3个身份, 60张人脸)...")
    
    from core.clustering_engine import DGFCClusteringEngine, DGFCConfig
    
    # 生成模拟数据: 3个簇的人脸特征
    np.random.seed(42)
    
    # 模拟3个身份的人脸特征 (512维向量)
    center1 = np.random.randn(512)
    center2 = np.random.randn(512)
    center3 = np.random.randn(512)
    
    cluster1 = np.random.randn(20, 512) * 0.15 + center1  # 身份1: 20张
    cluster2 = np.random.randn(25, 512) * 0.15 + center2  # 身份2: 25张  
    cluster3 = np.random.randn(15, 512) * 0.15 + center3  # 身份3: 15张
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([1]*20 + [2]*25 + [3]*15)
    
    # L2归一化
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"    数据维度: {embeddings.shape}")
    print(f"    真实标签分布: 身份1={20}张, 身份2={25}张, 身份3={15}张")
    
    # 运行DGFC聚类
    print("\n[2] 运行DGFC聚类...")
    clustering_engine = DGFCClusteringEngine(DGFCConfig(
        knn_k=15,
        tau=0.8,
        verbose=False,
    ))
    
    result = clustering_engine.fit_predict(embeddings)
    
    print(f"\n[3] 聚类结果:")
    print(f"    ✓ 发现簇数量: {result.n_clusters}")
    print(f"    ✓ 核心点数量: {np.sum(result.core_mask)} / {len(embeddings)}")
    print(f"    ✓ 簇大小分布: {[len(indices) for indices in result.cluster_indices.values()]}")
    
    # 计算评估指标
    print("\n[4] 评估聚类质量...")
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(true_labels, result.labels)
        nmi = normalized_mutual_info_score(true_labels, result.labels)
        
        print(f"    ✓ 调整兰德指数 (ARI): {ari:.4f}")
        print(f"    ✓ 归一化互信息 (NMI): {nmi:.4f}")
        
        if ari > 0.8:
            print("    ★ 聚类效果优秀!")
        elif ari > 0.5:
            print("    ★ 聚类效果良好")
        else:
            print("    ★ 聚类效果一般")
            
    except ImportError:
        print("    (需要sklearn计算评估指标)")
    
    # 可视化密度特征空间
    print("\n[5] 密度特征空间分析:")
    if result.density_features is not None:
        gsdata = result.density_features
        print(f"    特征空间范围: X=[{gsdata[:,0].min():.3f}, {gsdata[:,0].max():.3f}]")
        print(f"                  Y=[{gsdata[:,1].min():.3f}, {gsdata[:,1].max():.3f}]")
        print(f"    核心点密度分数均值: {result.density_scores[result.core_mask].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成! DGFC聚类算法运行正常")
    print("=" * 60)
    print("\n提示:")
    print("  - 运行 'python main.py' 启动Web服务 (需要安装完整依赖)")
    print("  - 运行 'python main.py --cli' 进入命令行模式")
    print("")


def run_camera():
    """运行摄像头实时识别"""
    logger.info("启动摄像头实时识别...")
    
    try:
        from core.recognition_engine import RecognitionEngine, RealtimeRecognizer
        
        engine = RecognitionEngine(
            detector_type=config.detection.detector_type,
            feature_model=config.feature.model_type,
            device=config.feature.device,
        )
        
        recognizer = RealtimeRecognizer(engine)
        recognizer.run_camera()
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
    except Exception as e:
        logger.error(f"摄像头启动失败: {e}")


def run_cli():
    """命令行交互模式"""
    import cmd
    
    class FaceRecognitionCLI(cmd.Cmd):
        intro = """
╔══════════════════════════════════════════════════════════╗
║          DGFC人脸识别系统 - 命令行模式                    ║
║    输入 'help' 查看可用命令, 'quit' 退出                  ║
╚══════════════════════════════════════════════════════════╝
        """
        prompt = 'DGFC> '
        
        def __init__(self):
            super().__init__()
            self.engine = None
        
        def do_init(self, arg):
            """初始化识别引擎"""
            print("正在初始化引擎...")
            try:
                from core.recognition_engine import RecognitionEngine
                self.engine = RecognitionEngine(
                    detector_type=config.detection.detector_type,
                    feature_model=config.feature.model_type,
                    device=config.feature.device,
                )
                print("✓ 引擎初始化成功")
            except Exception as e:
                print(f"✗ 初始化失败: {e}")
        
        def do_detect(self, arg):
            """检测图像中的人脸: detect <image_path>"""
            if not self.engine:
                print("请先运行 'init' 初始化引擎")
                return
            
            if not arg:
                print("用法: detect <image_path>")
                return
            
            import cv2
            image = cv2.imread(arg)
            if image is None:
                print(f"无法读取图像: {arg}")
                return
            
            faces = self.engine.detector.detect(image)
            print(f"检测到 {len(faces)} 张人脸")
            for i, face in enumerate(faces):
                print(f"  [{i+1}] bbox={face.bbox}, confidence={face.confidence:.3f}")
        
        def do_recognize(self, arg):
            """识别图像中的人脸: recognize <image_path>"""
            if not self.engine:
                print("请先运行 'init' 初始化引擎")
                return
            
            if not arg:
                print("用法: recognize <image_path>")
                return
            
            results = self.engine.process_image(arg)
            print(f"识别结果: {len(results)} 张人脸")
            for i, result in enumerate(results):
                name = result.person.name if result.person else "Unknown"
                print(f"  [{i+1}] {name} (confidence={result.confidence:.3f})")
        
        def do_register(self, arg):
            """注册新人员: register <name> <image_path1> [image_path2] ..."""
            if not self.engine:
                print("请先运行 'init' 初始化引擎")
                return
            
            args = arg.split()
            if len(args) < 2:
                print("用法: register <name> <image_path1> [image_path2] ...")
                return
            
            name = args[0]
            images = args[1:]
            
            person = self.engine.register_person(name, images)
            if person:
                print(f"✓ 已注册: {name} ({len(person.embeddings)} 张人脸)")
            else:
                print("✗ 注册失败: 未检测到有效人脸")
        
        def do_stats(self, arg):
            """显示系统统计信息"""
            if not self.engine:
                print("请先运行 'init' 初始化引擎")
                return
            
            print(f"注册人员数: {self.engine.database.num_persons}")
            print(f"人脸样本数: {self.engine.database.num_faces}")
        
        def do_quit(self, arg):
            """退出程序"""
            print("再见!")
            return True
        
        def do_exit(self, arg):
            """退出程序"""
            return self.do_quit(arg)
    
    FaceRecognitionCLI().cmdloop()


def main():
    parser = argparse.ArgumentParser(
        description="DGFC人脸识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # 启动Web服务
  python main.py --demo       # 运行演示
  python main.py --cli        # 命令行模式
  python main.py --camera     # 摄像头实时识别
        """
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='运行演示程序'
    )
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='命令行交互模式'
    )
    parser.add_argument(
        '--camera', 
        action='store_true',
        help='摄像头实时识别'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=config.web.host,
        help='Web服务器地址'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=config.web.port,
        help='Web服务器端口'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=config.feature.device,
        help='运行设备'
    )
    
    args = parser.parse_args()
    
    # 初始化目录
    init_directories()
    
    # 更新配置
    config.feature.device = args.device
    config.web.host = args.host
    config.web.port = args.port
    
    # 运行相应模式
    if args.demo:
        run_demo()
    elif args.cli:
        run_cli()
    elif args.camera:
        run_camera()
    else:
        run_web_server()


if __name__ == "__main__":
    main()

