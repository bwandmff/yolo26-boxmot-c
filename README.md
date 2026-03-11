# YOLO26 + BoxMOT Linux C Implementation

基于 Ultralytics YOLO26 和 ByteTrack 的纯 C 语言多目标跟踪系统，专为 Linux 环境优化。

## 项目简介

这是一个高性能的 Linux C 语言实现，结合了：
- 🎯 **YOLO26** - Ultralytics 最新目标检测模型 (ONNX Runtime 推理)
- 🔄 **ByteTrack** - 高性能多目标跟踪算法
- 📍 **轨迹可视化** - 带方向箭头的历史轨迹显示

## 功能特性

- 纯 C 语言实现，无 Python 依赖
- 支持 ONNX Runtime 推理
- 实时目标检测与跟踪
- 历史轨迹显示（带运动方向箭头）
- 支持视频文件和摄像头输入
- 命令行参数配置

## 环境要求

### 系统依赖 (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libopencv-dev \
    libonnxruntime-dev
```

### 可选：Python 依赖（用于模型转换）

```bash
pip install ultralytics onnxruntime
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/bwandmff/yolo26-boxmot-c.git
cd yolo26-boxmot-c
```

### 2. 编译

```bash
make
```

### 3. 运行

```bash
# 使用测试视频
./yolo26_bytetrack --model models/yolo26n.onnx --source models/test_video.mp4 --output output.mp4

# 使用摄像头
./yolo26_bytetrack --model models/yolo26n.onnx --source 0 --show
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | ONNX 模型路径 | models/yolo26n.onnx |
| `--source` | 视频文件或摄像头 ID | 0 |
| `--output` | 输出视频路径 | output.mp4 |
| `--tracker` | 跟踪器类型 | bytetrack |
| `--conf` | 置信度阈值 | 0.5 |
| `--show` | 显示实时窗口 | 关闭 |
| `--trail-length` | 轨迹长度 | 30 |
| `--help` | 显示帮助 | - |

## 项目结构

```
yolo26-boxmot-c/
├── Makefile              # 编译脚本
├── README.md             # 本文件
├── models/
│   ├── yolo26n.onnx     # YOLO26n ONNX 模型
│   ├── yolo26n.pt       # PyTorch 源模型
│   └── test_video.mp4   # 测试视频
├── include/
│   ├── yolo26.h         # YOLO26 检测 API
│   ├── bytetrack.h      # ByteTrack 跟踪 API
│   └── utils.h          # 工具函数
└── src/
    ├── main.c           # 主程序
    ├── yolo26.c         # YOLO26 推理实现
    ├── bytetrack.c      # ByteTrack 算法实现
    └── utils.c          # OpenCV/工具实现
```

## 性能指标

| 指标 | 数值 |
|------|------|
| YOLO26n mAP | 40.9% |
| ByteTrack FPS | 720+ |
| 模型大小 | 9.5 MB |

## 技术细节

### YOLO26 模型

YOLO26 是 Ultralytics 最新的端到端目标检测模型：
- 无需 NMS 后处理
- CPU 推理速度提升 43%
- 支持实例分割、姿态估计等任务

### ByteTrack 算法

ByteTrack 通过利用低分检测框实现更精确的跟踪：
- 两阶段关联策略
- 高分检测框优先匹配
- 低分检测框二次关联

## 编译选项

```bash
# Debug 模式
make debug

# 查看配置信息
make info

# 清理构建
make clean
```

## 常见问题

### 1. 编译报错：找不到 OpenCV

确保已安装 libopencv-dev：
```bash
sudo apt-get install libopencv-dev
```

### 2. 编译报错：找不到 ONNX Runtime

确保已安装 libonnxruntime-dev：
```bash
sudo apt-get install libonnxruntime-dev
```

### 3. 运行报错：无法打开视频

检查视频路径是否正确，或尝试使用摄像头 ID（0, 1, ...）

## 更新日志

### v1.0.0 (2026-03-11)
- 初始版本
- YOLO26n 模型支持
- ByteTrack 跟踪算法
- 轨迹可视化功能

## 许可证

MIT License

## 作者

[bwandmff](https://github.com/bwandmff)

## 参考项目

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
