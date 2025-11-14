"""测试脚本 - 使用改进的 Stage2FusionCA 模型
支持完整的生物识别指标评估：EER, AUC, TAR@FAR 等
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import warnings
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

# 导入模型和工具
from models.stage1 import EfficientViT, ConvNeXt
from models.stage2 import Stage2FusionCA
from utils.dataset import PairDataset
from utils.metrics import compute_eer, roc_auc, tar_at_far, far_frr_acc_at_threshold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 测试数据路径（修改为你的数据路径）
    palm_dir = 'C:/Users/admin/Desktop/palm_vein_fusion/data/PolyU/NIR'
    vein_dir = 'C:/Users/admin/Desktop/palm_vein_fusion/data/PolyU/Red'

    # 模型权重路径
    ckpt_path = 'outputs/models/stage2_best.pth'

    # 模型配置（需与训练时一致）
    num_classes = 100
    batch_size = 16

    # 是否保存 ROC 曲线图
    save_roc_curve = True
    output_dir = 'outputs/test_results'


config = Config()
os.makedirs(config.output_dir, exist_ok=True)

def get_transforms():
    """测试数据预处理（无数据增强）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 与训练时一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_models(config):
    """加载训练好的模型"""
    device = config.device

    # 创建模型
    vit_palm = EfficientViT(
        img_size=224, in_chans=1,
        embed_dim=[64, 128, 192],
        key_dim=[16, 16, 16],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4]
    ).to(device)

    vit_vein = EfficientViT(
        img_size=224, in_chans=1,
        embed_dim=[64, 128, 192],
        key_dim=[16, 16, 16],
        depth=[1, 2, 3],
        num_heads=[4, 4, 4]
    ).to(device)

    cnn_palm = ConvNeXt(
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ).to(device)

    cnn_vein = ConvNeXt(
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ).to(device)

    fusion_model = Stage2FusionCA(
        in_dim_global_palm=192,
        in_dim_global_vein=192,
        in_dim_local_palm=768,
        in_dim_local_vein=768,
        out_dim_global=256,
        out_dim_local=256,
        use_spatial_fusion=True,
        final_l2norm=True,
        with_arcface=False,  # 测试时不需要 ArcFace
        num_classes=config.num_classes
    ).to(device)

    # 加载权重
    if os.path.exists(config.ckpt_path):
        print(f"Loading checkpoint from {config.ckpt_path}...")
        ckpt = torch.load(config.ckpt_path, map_location=device)

        vit_palm.load_state_dict(ckpt['vit_palm'])
        vit_vein.load_state_dict(ckpt['vit_vein'])
        cnn_palm.load_state_dict(ckpt['cnn_palm'])
        cnn_vein.load_state_dict(ckpt['cnn_vein'])
        fusion_model.load_state_dict(ckpt['fusion'])

        print(f"✓ Checkpoint loaded (Best Acc: {ckpt.get('best_acc', 'N/A')})")
    else:
        print(f"⚠ Warning: Checkpoint not found at {config.ckpt_path}")
        print("Using random initialization (for testing only)")

    # 设置为评估模式
    for model in [vit_palm, vit_vein, cnn_palm, cnn_vein, fusion_model]:
        model.eval()

    return vit_palm, vit_vein, cnn_palm, cnn_vein, fusion_model


def extract_features(vit_palm, vit_vein, cnn_palm, cnn_vein, fusion_model, palm_img, vein_img, device):
    """提取融合特征向量"""
    with torch.no_grad():
        # Stage1: 特征提取
        palm_global = vit_palm(palm_img, pool=True)  # (B, 192)
        vein_global = vit_vein(vein_img, pool=True)  # (B, 192)
        palm_local = cnn_palm(palm_img, return_spatial=True)  # (B, 768, H, W)
        vein_local = cnn_vein(vein_img, return_spatial=True)  # (B, 768, H, W)

        # Stage2: 融合（不需要 labels）
        fused_feat, _ = fusion_model(
            palm_global, vein_global,
            palm_local, vein_local
        )

    return fused_feat.cpu().numpy()

def compute_verification_scores(all_features, all_labels):
    """
    计算验证分数（genuine 和 impostor）
    genuine: 同一身份的样本对
    impostor: 不同身份的样本对
    """
    genuine_scores = []
    impostor_scores = []

    # 按身份分组
    id_to_indices = {}
    for idx, label in enumerate(all_labels):
        if label not in id_to_indices:
            id_to_indices[label] = []
        id_to_indices[label].append(idx)

    print(f"\n身份统计: {len(id_to_indices)} 个不同身份")

    # 计算 genuine scores (同一身份的所有配对)
    for label, indices in tqdm(id_to_indices.items(), desc="计算 Genuine Scores"):
        if len(indices) < 2:
            continue
        for i, j in itertools.combinations(indices, 2):
            sim = cosine_similarity([all_features[i]], [all_features[j]])[0][0]
            genuine_scores.append(sim)

    # 计算 impostor scores (不同身份的配对)
    # 为了避免组合爆炸，每个身份只与部分其他身份配对
    all_identities = list(id_to_indices.keys())
    for i, label in enumerate(tqdm(all_identities, desc="计算 Impostor Scores")):
        # 与最多 10 个其他身份配对
        other_labels = all_identities[i+1:i+11]
        for other_label in other_labels:
            # 每个身份取最多 3 个样本
            for idx1 in id_to_indices[label][:3]:
                for idx2 in id_to_indices[other_label][:3]:
                    sim = cosine_similarity([all_features[idx1]], [all_features[idx2]])[0][0]
                    impostor_scores.append(sim)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    return genuine_scores, impostor_scores


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """绘制 ROC 曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")


def plot_score_distribution(genuine_scores, impostor_scores, save_path):
    """绘制分数分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='green', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red', density=True)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distribution: Genuine vs Impostor')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Score distribution saved to {save_path}")


def main():
    print("=" * 70)
    print("掌纹掌静脉融合识别 - 测试脚本")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Checkpoint: {config.ckpt_path}")
    print("=" * 70)

    # 1. 加载模型
    print("\n[步骤 1/5] 加载模型...")
    vit_palm, vit_vein, cnn_palm, cnn_vein, fusion_model = load_models(config)

    # 2. 加载测试数据
    print("\n[步骤 2/5] 加载测试数据...")
    test_dataset = PairDataset(
        config.palm_dir, config.vein_dir,
        get_transforms(),
        split='val'  # 使用验证集作为测试集
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"✓ Test dataset size: {len(test_dataset)}")

    # 3. 提取特征
    print("\n[步骤 3/5] 提取特征向量...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for palm, vein, labels in tqdm(test_loader, desc="Extracting Features"):
            palm = palm.to(config.device)
            vein = vein.to(config.device)

            features = extract_features(
                vit_palm, vit_vein, cnn_palm, cnn_vein, fusion_model,
                palm, vein, config.device
            )

            all_features.append(features)
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"✓ Extracted {len(all_features)} feature vectors")

    # 4. 计算验证分数
    print("\n[步骤 4/5] 计算验证分数...")
    genuine_scores, impostor_scores = compute_verification_scores(all_features, all_labels)

    print(f"\n分数统计:")
    print(f"  Genuine 样本对: {len(genuine_scores)}")
    print(f"  Impostor 样本对: {len(impostor_scores)}")
    print(f"  Genuine 均值: {np.mean(genuine_scores):.4f} ± {np.std(genuine_scores):.4f}")
    print(f"  Impostor 均值: {np.mean(impostor_scores):.4f} ± {np.std(impostor_scores):.4f}")

    # 5. 计算生物识别指标
    print("\n[步骤 5/5] 计算生物识别指标...")

    # 合并标签和分数
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ]).astype(int)

    # 计算 EER
    eer, eer_threshold = compute_eer(scores, labels, is_similarity=True, return_threshold=True)

    # 计算 EER 阈值处的 FAR, FRR, ACC
    eer_metrics = far_frr_acc_at_threshold(scores, labels, eer_threshold, is_similarity=True)

    # 计算 ROC 和 AUC
    fpr, tpr, thresholds, auc_score = roc_auc(scores, labels, is_similarity=True)

    # 计算不同 FAR 下的 TAR
    target_fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    tar_results = []
    for far in target_fars:
        res = tar_at_far(scores, labels, target_far=far, is_similarity=True)
        tar_results.append(res)

    # ============================================
    # 输出结果
    # ============================================
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    print(f"\n【核心指标】")
    print(f"  EER (Equal Error Rate):        {eer:.4f} ({eer*100:.2f}%)")
    print(f"  EER Threshold:                 {eer_threshold:.4f}")
    print(f"  AUC (Area Under Curve):        {auc_score:.4f}")

    print(f"\n【EER 阈值处的性能】")
    print(f"  FAR (False Accept Rate):       {eer_metrics['FAR']:.4f}")
    print(f"  FRR (False Reject Rate):       {eer_metrics['FRR']:.4f}")
    print(f"  ACC (Accuracy):                {eer_metrics['ACC']:.4f} ({eer_metrics['ACC']*100:.2f}%)")

    print(f"\n【不同 FAR 下的 TAR (True Accept Rate)】")
    print(f"{'FAR':<12} | {'TAR':<12} | {'Threshold':<12}")
    print("-" * 40)
    for far, res in zip(target_fars, tar_results):
        thr_str = f"{res['threshold']:.4f}" if np.isfinite(res['threshold']) else 'inf'
        print(f"{far:<12.5f} | {res['TAR']:<12.4f} | {thr_str:<12}")

    # ============================================
    # 保存可视化结果
    # ============================================
    if config.save_roc_curve:
        print(f"\n保存可视化结果到 {config.output_dir}/...")
        plot_roc_curve(fpr, tpr, auc_score, os.path.join(config.output_dir, 'roc_curve.png'))
        plot_score_distribution(genuine_scores, impostor_scores,
                               os.path.join(config.output_dir, 'score_distribution.png'))

    # ============================================
    # 保存详细结果到文件
    # ============================================
    result_file = os.path.join(config.output_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("掌纹掌静脉融合识别 - 测试结果\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Checkpoint: {config.ckpt_path}\n")
        f.write(f"Test samples: {len(all_labels)}\n")
        f.write(f"Genuine pairs: {len(genuine_scores)}\n")
        f.write(f"Impostor pairs: {len(impostor_scores)}\n\n")

        f.write("【核心指标】\n")
        f.write(f"EER: {eer:.6f}\n")
        f.write(f"AUC: {auc_score:.6f}\n\n")

        f.write("【TAR @ FAR】\n")
        f.write(f"{'FAR':<12} | {'TAR':<12} | {'Threshold':<12}\n")
        f.write("-" * 40 + "\n")
        for far, res in zip(target_fars, tar_results):
            thr_str = f"{res['threshold']:.4f}" if np.isfinite(res['threshold']) else 'inf'
            f.write(f"{far:<12.5f} | {res['TAR']:<12.4f} | {thr_str:<12}\n")

    print(f"\n✓ 详细结果已保存到: {result_file}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()