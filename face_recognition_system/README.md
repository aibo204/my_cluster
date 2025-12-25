# DGFC人脸识别系统

基于**密度聚类算法(DGFC)**的智能人脸识别系统，适合作为毕业设计项目。

## 📋 系统概述

本系统集成了自主设计的DGFC（Density-GMM Flood Clustering）聚类算法，实现了完整的人脸识别解决方案，包括：

- 🔍 **人脸检测** - 基于MTCNN的精确人脸检测与对齐
- 🧬 **特征提取** - 使用FaceNet/ArcFace深度学习模型提取512维人脸特征
- 🔗 **智能聚类** - 原创DGFC算法实现自动人脸分组
- 🎯 **人脸识别** - 1:N人脸检索与身份识别
- 📹 **实时监控** - 摄像头实时人脸识别
- 🌐 **Web界面** - 现代化响应式管理界面

## 🏗️ 系统架构

```
face_recognition_system/
├── config/                 # 系统配置
│   ├── __init__.py
│   └── settings.py         # 配置参数定义
├── core/                   # 核心算法模块
│   ├── __init__.py
│   ├── detector.py         # 人脸检测器 (MTCNN)
│   ├── feature_extractor.py # 特征提取 (FaceNet/ArcFace)
│   ├── clustering_engine.py # DGFC聚类引擎
│   └── recognition_engine.py # 识别引擎
├── database/               # 数据库模块
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy数据模型
│   └── face_db.py          # 人脸数据库管理+FAISS索引
├── web/                    # Web服务
│   ├── __init__.py
│   ├── app.py              # Flask应用
│   ├── templates/          # HTML模板
│   │   └── index.html
│   └── static/             # 静态资源
│       ├── css/style.css
│       └── js/app.js
├── data/                   # 数据目录
│   ├── faces/              # 人脸图像存储
│   ├── originals/          # 原始图像存储
│   └── faces.db            # SQLite数据库
├── logs/                   # 日志目录
├── main.py                 # 主入口
├── requirements.txt        # Python依赖
└── README.md               # 项目文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行系统

```bash
# 启动Web服务
python main.py

# 运行演示
python main.py --demo

# 命令行模式
python main.py --cli

# 摄像头实时识别
python main.py --camera
```

访问 http://localhost:5000 进入Web管理界面。

## 💡 核心算法：DGFC

DGFC（Density-GMM Flood Clustering）是本项目的核心创新点，针对高维人脸特征向量设计的聚类算法：

### 算法流程

1. **构建kNN图** - 使用余弦距离建立k近邻关系
2. **计算密度特征** - 提取局部方差和中位距离的倒数
3. **密度特征空间** - 归一化后构建2D密度特征空间
4. **核心点检测** - 使用2-GMM识别高密度核心点
5. **洪泛填充聚类** - 从高密度点开始扩展聚类
6. **DSU合并** - 使用并查集合并碰撞的簇
7. **剩余点分配** - 将非核心点分配到最近的簇

### 算法优势

- ✅ 无需预设簇数量
- ✅ 能处理任意形状的簇
- ✅ 对噪声点鲁棒
- ✅ 自适应局部密度

## 🔧 API接口

### 人脸检测
```
POST /api/detect
Content-Type: application/json

{
    "image": "<base64_encoded_image>"
}
```

### 人脸识别
```
POST /api/recognize
Content-Type: application/json

{
    "image": "<base64_encoded_image>"
}
```

### 人员注册
```
POST /api/register
Content-Type: application/json

{
    "name": "张三",
    "images": ["<base64_image1>", "<base64_image2>"]
}
```

### 批量聚类
```
POST /api/cluster
Content-Type: application/json

{
    "images": ["<base64_image1>", ...],
    "auto_register": true
}
```

### 人脸比对
```
POST /api/compare
Content-Type: application/json

{
    "image1": "<base64_image1>",
    "image2": "<base64_image2>"
}
```

## 📊 技术栈

| 模块 | 技术选型 |
|------|----------|
| 人脸检测 | MTCNN, OpenCV |
| 特征提取 | FaceNet (InceptionResnetV1) |
| 聚类算法 | DGFC (原创) |
| 向量索引 | FAISS |
| 数据库 | SQLite + SQLAlchemy |
| Web框架 | Flask + SocketIO |
| 前端 | HTML5 + CSS3 + JavaScript |

## 📈 工作量说明

本项目作为毕业设计，包含以下工作量：

### 算法设计与实现
- [x] DGFC聚类算法设计与数学推导
- [x] kNN图构建与优化
- [x] GMM核心点检测
- [x] 洪泛填充聚类实现
- [x] 并查集簇合并

### 系统开发
- [x] 人脸检测模块 (MTCNN集成)
- [x] 人脸对齐与预处理
- [x] 深度特征提取 (FaceNet)
- [x] 识别引擎实现
- [x] 实时视频处理

### 数据库设计
- [x] 数据模型设计 (Person, Face, Cluster)
- [x] SQLite数据库实现
- [x] FAISS向量索引集成

### Web开发
- [x] Flask后端API (10+ 接口)
- [x] WebSocket实时通信
- [x] 响应式前端界面
- [x] 文件上传与处理

### 文档与测试
- [x] 系统架构文档
- [x] API接口文档
- [x] 演示程序

## 🔬 实验评估建议

1. **聚类性能评估**
   - 在LFW、YTF等标准数据集上测试
   - 计算ARI、NMI等聚类指标
   - 与DBSCAN、K-Means等算法对比

2. **识别性能评估**
   - 计算识别准确率、召回率
   - 绘制ROC曲线
   - 测试不同阈值的影响

3. **效率评估**
   - 测试不同数据规模下的处理时间
   - 对比CPU和GPU性能

## 📝 论文撰写建议

1. **绪论**：人脸识别背景、研究意义、国内外研究现状
2. **相关技术**：深度学习、聚类算法、人脸检测技术
3. **DGFC算法**：算法设计、数学推导、伪代码
4. **系统设计**：架构设计、模块设计、数据库设计
5. **系统实现**：关键代码、技术难点、解决方案
6. **实验评估**：实验设计、结果分析、对比实验
7. **总结展望**：研究成果、创新点、未来工作

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- FaceNet论文和实现
- MTCNN人脸检测
- FAISS向量索引库
- Flask Web框架

---

**作者**: [Your Name]  
**指导教师**: [Advisor Name]  
**完成时间**: 2025年

如有问题，请联系 [your-email@example.com]

