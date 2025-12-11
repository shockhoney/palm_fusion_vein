import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 直接复用 train.py 里的配置和工具函数
from train_teacher import config, build_backbone, create_phase2_dataloaders, EarlyStopping
from models.stage2 import Stage2Fusion
from utils.head import Arcface_Head


def build_stage2_teacher():
    """
    从 outputs/models/stage2_best.pth 构建并加载 Teacher:
    - cnn_palm_T: Edgenext 掌纹
    - cnn_vein_T: Edgenext 掌静脉
    - fusion_T : Stage2Fusion
    """
    # Edgenext backbone 作为 teacher
    cnn_palm_T, feat_dim_T, _ = build_backbone('edgenext')
    cnn_vein_T, _, _           = build_backbone('edgenext')

    fusion_T = Stage2Fusion(
        in_dim_global=feat_dim_T,
        out_dim_final=512,
        final_l2norm=True
    ).to(config.device)

    ckpt_path = os.path.join(config.save_dir, 'stage2_best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=config.device)
    cnn_palm_T.load_state_dict(ckpt['cnn_palm'])
    cnn_vein_T.load_state_dict(ckpt['cnn_vein'])
    fusion_T.load_state_dict(ckpt['fusion'])

    # 冻结 teacher
    for m in [cnn_palm_T, cnn_vein_T, fusion_T]:
        m.to(config.device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    print(f"==> Loaded teacher from {ckpt_path}")
    return cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T


def train_joint_distill(log_dir='runs_distill'):
    """
    两阶段联合蒸馏：
    Teacher: Edgenext + Stage2Fusion (from stage2_best.pth)
    Student: MobileFaceNet + Stage2Fusion + ArcFace Head
    """
    writer = SummaryWriter(log_dir=log_dir)

    # ---------- 1. Teacher ----------
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher()

    # ---------- 2. Student ----------
    # 两个 MobileFaceNet 分支
    cnn_palm_S, feat_dim_S, _ = build_backbone('mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('mobilefacenet')

    # 学生 Stage2 融合
    fusion_S = Stage2Fusion(
        in_dim_global=feat_dim_S,
        out_dim_final=512,
        final_l2norm=True
    ).to(config.device)

    # Phase2 成对 dataloader
    train_loader, val_loader, num_classes = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch
    )

    # 学生分类头（和原 Stage2 保持一致超参）
    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=30.0,
        m=0.20,
    ).to(config.device)

    # ---------- 3. 优化器 / scheduler ----------
    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.p2_epochs
    )

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    beta_kd = 1.0    # 融合特征蒸馏权重，可以根据效果调整

    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0

    for epoch in range(config.p2_epochs):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[JointDistill] Epoch {epoch+1}/{config.p2_epochs}',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device)
            vein_img = vein_img.to(config.device)
            labels   = labels.to(config.device)

            # ---------- Teacher 前向（无梯度） ----------
            with torch.no_grad():
                F_palm_T = cnn_palm_T(palm_img, return_spatial=False)   # [B, feat_dim_T]
                F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
                fused_T  = fusion_T(F_palm_T, F_vein_T)                # [B, 512]
                fused_T_n = F.normalize(fused_T, p=2, dim=1)

            # ---------- Student 前向 ----------
            F_palm_S = cnn_palm_S(palm_img, return_spatial=False)      # [B, feat_dim_S]
            F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
            fused_S  = fusion_S(F_palm_S, F_vein_S)                    # [B, 512]
            fused_S_n = F.normalize(fused_S, p=2, dim=1)

            logits_S = classifier_S(fused_S, labels)

            # ---------- Loss ----------
            # 1) 分类 loss
            loss_ce = ce_loss(logits_S, labels)

            # 2) 融合特征蒸馏 loss
            loss_kd = mse_loss(fused_S_n, fused_T_n)

            loss = loss_ce + beta_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_palm_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier_S.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(logits_S, 1)
            train_correct += (pred == labels).sum().item()
            train_total   += labels.size(0)

            pbar.update(1)

        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100. * train_correct / train_total

        # ---------- 验证（只看学生） ----------
        cnn_palm_S.eval()
        cnn_vein_S.eval()
        fusion_S.eval()
        classifier_S.eval()

        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device)
                vein_img = vein_img.to(config.device)
                labels   = labels.to(config.device)

                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)
                logits_S = classifier_S(fused_S, labels)

                loss = ce_loss(logits_S, labels)
                val_total_loss += loss.item()
                _, pred = torch.max(logits_S, 1)
                val_correct += (pred == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_acc  = 100. * val_correct / val_total

        print(f"[JointDistill][Epoch {epoch+1}] "
              f"TrLoss={avg_train_loss:.4f}, TrAcc={avg_train_acc:.2f}%, "
              f"VaLoss={avg_val_loss:.4f}, VaAcc={avg_val_acc:.2f}%")

        # 写 tensorboard
        writer.add_scalar('JointDistill/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('JointDistill/TrainAcc',  avg_train_acc,  epoch)
        writer.add_scalar('JointDistill/ValLoss',   avg_val_loss,   epoch)
        writer.add_scalar('JointDistill/ValAcc',    avg_val_acc,    epoch)

        # 保存最优学生模型
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'cnn_palm_S': cnn_palm_S.state_dict(),
                'cnn_vein_S': cnn_vein_S.state_dict(),
                'fusion_S': fusion_S.state_dict(),
                'classifier_S': classifier_S.state_dict(),
            }, os.path.join(config.save_dir, 'stage2_student_joint_best.pth'))
            print(f"  >>> New best student model saved. ValAcc={best_acc:.2f}%")

        # 提前停止
        if early_stop(-avg_val_acc, mode='min'):
            print(f"[JointDistill] Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    return best_acc


if __name__ == '__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    best = train_joint_distill()
    print(f"[JointDistill] Final best val acc: {best:.2f}%")
