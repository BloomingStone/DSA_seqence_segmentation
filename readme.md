# 深度学习实验项目

本项目是一个基于PyTorch的深度学习实验框架，专注于图像处理相关任务。包含基础模型实现、实验管理模块和训练/预测工具。

## 项目结构

```
├── .vscode/                 # VSCode调试配置
│   └── launch.json          # 实验模块调试配置
├── src/                     # 源代码目录
│   ├── core/                # 核心模块
│   │   ├── base_models/     # 基础模型实现
│   │   │   ├── ResEncUNet_pure_torch.py  # 纯PyTorch版ResEncUNet
│   │   │   └── ResEncUNet.py             # ResEncUNet模型
│   │   └── utils/           # 工具类
│   │       ├── log.py       # 日志工具
│   │       ├── metrics.py   # 评估指标
│   │       └── tensorboard.py # TensorBoard集成
│   └── experiments/         # 实验模块
│       ├── baseline_pure_torch/  # 纯PyTorch基线实验
│       │   ├── dataloader.py     # 数据加载器
│       │   ├── single_epochs.py  # 单epoch训练函数
|       │   ├── image_type.py     # 实验涉及到的图像类型（主要用于monai中）
│       │   ├── train.py     # 训练脚本
│       │   ├── train_sub.py # 训练子实验，此处是简化版UNet
│       │   ├── export_onnx.py # ONNX导出
|       │   ├── export_onnx_sub.py  # 子实验结果导出onnx
│       │   ├── only_test.py   # 测试脚本
│       │   └── params.toml  # 参数配置
|       |   └── sub_experiments/   # 简化版UNet子实验(有多个)
│       └── first_frame_assist/   # 第一帧辅助实验
│           ├── model.py     # 模型定义
│           ├── train.py     # 训练脚本
│           ├── predict.py   # 预测脚本
│           └── export_onnx.py # ONNX导出
└── readme.md                # 项目说明
```

## 快速开始

使用VSCode调试配置（`.vscode/launch.json`）可快速运行实验模块，实验细节设置可以在各实验文件夹下`params.toml`中修改。

1. **第一帧辅助实验预测**  
   `first_frame_assist.predict` - 运行预测任务

2. **导出ONNX模型**  
   `first_frame_assist.export_onnx` - 导出ONNX格式模型

3. **纯PyTorch基线训练**  
   `baseline_pure_torch.train` - 训练基线模型

4. **简化UNet训练**  
   `simplified_unet——train` - 训练简化版UNet模型

## 一般训练

可以使用 `python -m` 调用具体的模块，如 `python -m src.experiments.first_frame_assist.predict` 

## 实验管理

每个实验包含：
- 训练脚本（train.py）
- 预测脚本（predict.py）
- ONNX导出脚本（export_onnx.py）
- 参数配置（params.toml）
- 输出目录：
  - checkpoints/：模型检查点
  - log/：训练日志
  - tensorboard/：可视化日志
