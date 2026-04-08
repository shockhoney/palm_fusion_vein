# 掌纹掌静脉融合识别系统

基于深度学习的轻量化多模态生物特征融合识别系统，采用知识蒸馏技术实现模型压缩与加速。

## 研究背景

本项目面向生物特征识别领域，融合掌纹(Palmprint)与掌静脉(Palm Vein)两种模态信息，通过跨模态注意力机制和通道注意力融合策略，提升识别精度与鲁棒性。系统采用两阶段训练策略，并引入知识蒸馏技术将大型教师模型的知识迁移至轻量级学生模型，实现精度与效率的平衡。

## 方法概述

### 网络架构


### 训练策略

**阶段一：单模态预训练**
- 分别训练掌纹、掌静脉特征提取网络
- 使用ArcFace Loss进行度量学习

**阶段二：融合模型训练**
- 加载预训练的单模态网络
- 引入跨模态注意力机制
- 端到端微调融合网络

**知识蒸馏**
- 教师模型：MobileFaceNet + Stage2Fusion
- 学生模型：TinyMobileFaceNet + StudentFusion
- 蒸馏损失：Embedding KD + Relational KD + Classification Loss

### 核心模块

| 模块 | 描述 |
|------|------|
| MobileFaceNet | 教师骨干网络，基于深度可分离卷积 |
| TinyMobileFaceNet | 学生骨干网络，更少的通道数和残差块，附加ECA注意力 |
| CrossModalAttention | 跨模态注意力机制，增强模态间信息交互 |
| ChannelAttentionFusion | 通道注意力融合，自适应学习模态权重 |
| Stage2FusionStudent_BottleneckGate | 学生融合模块，采用瓶颈结构与门控机制 |


```

## 环境配置

### 依赖要求

- Python 3.8+
- PyTorch 1.10+ (支持CUDA)
- 其他依赖见 `requirements.txt`

### 安装

```bash
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

## 使用方法

### 数据准备

数据集列表文件格式 (`txt-datasets/`)：

**单模态列表** (用于阶段一)：
```
/path/to/image1.jpg label1
/path/to/image2.jpg label2
...
```

**配对列表** (用于阶段二)：
```
/path/to/palm1.jpg /path/to/vein1.jpg label1
/path/to/palm2.jpg /path/to/vein2.jpg label2
...
```

### 训练教师模型

```bash
python train_teacher.py
```

训练参数配置位于脚本内Config类：

### 知识蒸馏训练

```bash
python train_s_vkd.py \
    --train_list txt-datasets/polyu_phase2_train.txt \
    --val_list txt-datasets/polyu_phase2_val.txt \
    --teacher_ckpt outputs/models/stage2_best.pth \
    --epochs 200 \
    --batch_size 16 \
    --lambda_emb 2.0 \
    --lambda_rel 2.0
```

### 模型测试

```bash
python test.py
```

测试配置（位于脚本内Config类）：


## 引用

如果本项目对您的研究有帮助，请引用相关论文。
