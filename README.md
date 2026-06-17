# Palmprint--Palm-Vein Fusion Verification

本项目用于掌纹/掌静脉双模态验证。当前修订版采用 **ResNet18 教师网络** 和 **MobileFaceNet+ECA 学生网络**：

- Teacher：双分支 ResNet18 encoder + Stage2Fusion + ArcFace
- Student：双分支 MobileFaceNet encoder（内置 ECA）+ Bottleneck-Gated Fusion + ArcFace
- Distillation：classification loss + embedding KD + relational KD + teacher-confidence weighting + ramp-up

## 环境

推荐使用已有 conda 环境 `pvf`：

```powershell
conda activate pvf
```

如果缺依赖，可安装：

```powershell
pip install -r requirements.txt
```

PyTorch 建议按本机 CUDA 版本单独安装。例如 CUDA 11.8：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 数据准备

原始数据应放在：

```text
data/
  CASIA/
  CUMT/
  PolyU/
  tongji/
```

生成训练、验证、测试列表：

```powershell
python prepare_data_txt.py
```

输出位于：

```text
data_txt/
```

当前协议为身份不重叠开集测试，训练:测试 = 8:2。验证集从训练身份中划分，仅用于 checkpoint 选择和融合权重选择。

查看协议统计：

```powershell
python utils\protocol_stats.py `
  --pair-list PolyU:train=data_txt/polyu_phase2_train.txt `
  --pair-list PolyU:val=data_txt/polyu_phase2_val.txt `
  --pair-list PolyU:test=data_txt/polyu_phase2_test.txt
```

## 预训练权重

ResNet18 教师网络默认使用本地 ImageNet 预训练权重：

```text
pretrain/resnet18_imagenet1k_v1.pth
```

如需从零训练，可传入空路径：

```powershell
--pretrained_path ""
```

## 训练 ResNet18 教师网络

以 PolyU 为例：

```powershell
python train_teacher.py `
  --backbone resnet18 `
  --pretrained_path pretrain/resnet18_imagenet1k_v1.pth `
  --list_file_palm data_txt/PolyU_palmprint_list.txt `
  --list_file_vein data_txt/PolyU_palmvein_list.txt `
  --phase2_train data_txt/polyu_phase2_train.txt `
  --phase2_val data_txt/polyu_phase2_val.txt `
  --save_dir outputs/teacher_resnet18 `
  --run_name PolyU_seed42 `
  --seed 42
```

教师 checkpoint：

```text
outputs/teacher_resnet18/PolyU_seed42/stage2_best.pth
```

## 蒸馏 MobileFaceNet+ECA 学生网络

```powershell
python train_s_vkd.py `
  --train_list data_txt/polyu_phase2_train.txt `
  --val_list data_txt/polyu_phase2_val.txt `
  --teacher_ckpt outputs/teacher_resnet18/PolyU_seed42/stage2_best.pth `
  --save_dir outputs/student_mobilefacenet `
  --run_name PolyU_bs8_seed42 `
  --seed 42 `
  --batch_size 8
```

训练输出包括：

- `student_best_distill.pth`
- `student_last_distill.pth`
- `val_metrics.csv`
- `teacher_confidence_stats.csv`
- `teacher_confidence_last_epoch.npy`

## 多 seed 和 batch size 实验

```powershell
python scripts\run_sweeps.py `
  --dataset PolyU `
  --teacher_ckpt outputs/teacher_resnet18/PolyU_seed42/stage2_best.pth `
  --save_dir outputs/student_mobilefacenet `
  --seeds 42 43 44 `
  --batch_sizes 8 16 32
```

汇总 mean/std：

```powershell
python scripts\summarize_runs.py --root outputs/student_mobilefacenet
```

## 测试与 failure case 导出

```powershell
python test.py `
  --backbone mobilefacenet `
  --ckpt outputs/student_mobilefacenet/PolyU_bs8_seed42/student_best_distill.pth `
  --palm_list data_txt/PolyU_palmprint_list.txt `
  --vein_list data_txt/PolyU_palmvein_list.txt `
  --pair_txt data_txt/polyu_phase2_test.txt `
  --out_csv outputs/eval_polyu.csv `
  --failure_csv outputs/failure_polyu.csv
```

## 复杂度和延迟

CPU latency：

```powershell
python lightweight_metrics.py --model teacher --device cpu --warmup 50 --iters 200
python lightweight_metrics.py --model student --device cpu --warmup 50 --iters 200
```

GPU latency：

```powershell
python lightweight_metrics.py --model teacher --device cuda --warmup 50 --iters 200
python lightweight_metrics.py --model student --device cuda --warmup 50 --iters 200
```

## 教师置信度分布

```powershell
python scripts\plot_teacher_confidence.py `
  --npy outputs/student_mobilefacenet/PolyU_bs8_seed42/teacher_confidence_last_epoch.npy `
  --out outputs/student_mobilefacenet/PolyU_bs8_seed42/teacher_confidence_hist.png
```

## 主要文件

- `train_teacher.py`：训练 ResNet18 fusion teacher
- `train_s_vkd.py`：训练 MobileFaceNet+ECA distilled student
- `test.py`：评估 AUC、EER、TAR@FAR，并可导出 failure cases
- `lightweight_metrics.py`：统计参数量、FLOPs、模型大小和 latency
- `prepare_data_txt.py`：生成身份不重叠数据列表
- `utils/protocol_stats.py`：统计 split 和 genuine/impostor pair 数
