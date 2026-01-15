import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.student_fusion import Stage2FusionStudent_BottleneckGate    
from train_teacher import config, build_backbone, create_phase2_dataloaders, EarlyStopping
from models.stage2 import Stage2Fusion
from utils.head import Arcface_Head


def build_stage2_teacher():

    cnn_palm_T, feat_dim_T, _ = build_backbone('mobilefacenet')
    cnn_vein_T, _, _           = build_backbone('mobilefacenet')
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

def kd_cosine(s, t):
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()

def kd_cosine_per_sample(s, t):
    """Per-sample cosine distance (MoVE-KD uses per-sample weighting)."""
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1))  # [B]


def _move_token_weighted_mse_fuse(fused_S, fused_T, use_token_weight: bool):
    """
    MoVE-KD uses token-wise weighting based on CLIP attention.
    In this project we don't have patch tokens/attention, so we adapt the same idea by
    weighting *feature dimensions* of the fused embedding using the (normalized) teacher
    fused embedding magnitude as importance weights.

    - base term: mean MSE over dims
    - token-weight term: weighted sum over dims

    This mirrors MoVE-KD's pattern:
      teacher_loss.mean(...) + (teacher_loss * token_weight).sum(...)
    """
    s = F.normalize(fused_S, dim=1)
    t = F.normalize(fused_T, dim=1)
    mse_vec = (s - t).pow(2)  # [B, 512]
    base = mse_vec.mean(dim=1)  # [B]
    if not use_token_weight:
        return base

    # importance weights (softmax), analogous to CLS->token attention weights in MoVE-KD
    w = torch.softmax(t.abs(), dim=1)  # [B, 512]
    weighted = (mse_vec * w).sum(dim=1)  # [B]
    return base + weighted


def move_kd_loss_from_features(
    fused_S, fused_T,
    palm_S, palm_T,
    vein_S, vein_T,
    kd_w: float,
    kd_memory_w: float,
    use_token_weight: bool,
    use_teacher_weight: bool,
    teacher_temp: float = 1.0,
):
    """
    Minimal MoVE-KD-style distillation loss adapted to this project.

    In MoVE-KD (move_llava_llama.py):
      - teacher_loss: per-teacher per-sample loss aggregated from MSE(reduction='none')
      - token_weight: reweight important tokens
      - teacher_weight: fixed alpha for CLIP + softmax for other teachers
      - kd_loss added to task loss

    Here:
      - "main teacher" == fusion teacher (gets alpha mass)
      - "other teachers" == palm / vein branches (share 1-alpha)
      - token_weight is applied to fused embedding dimensions (proxy for patch tokens)

    Returns:
      kd_loss (scalar), debug dict.
    """
    loss_fuse = _move_token_weighted_mse_fuse(fused_S, fused_T, use_token_weight)  # [B]
    loss_palm = kd_cosine_per_sample(palm_S, palm_T)  # [B]
    loss_vein = kd_cosine_per_sample(vein_S, vein_T)  # [B]

    teacher_loss = torch.stack([loss_fuse, loss_palm, loss_vein], dim=1)  # [B, 3]

    b = teacher_loss.size(0)
    device = teacher_loss.device
    dtype = teacher_loss.dtype

    alpha = float(kd_memory_w)
    alpha = min(max(alpha, 0.0), 1.0)

    if use_teacher_weight:
        # MoVE-KD uses similarity-based weights; we approximate with negative loss as score.
        else_scores = torch.stack([-loss_palm, -loss_vein], dim=1)  # [B,2]
        else_w = torch.softmax(else_scores / max(1e-6, float(teacher_temp)), dim=1)  # [B,2]

        # MoVE-KD bias heuristic when one teacher dominates.
        else_w_mean = else_w.mean(dim=0)
        if else_w_mean.max() > 0.8:
            bias = 0.2
            else_w = torch.softmax((else_scores / max(1e-6, float(teacher_temp))) + bias, dim=1)

        main_w = torch.full((b, 1), alpha, device=device, dtype=dtype)
        tea_w = torch.cat([main_w, else_w * (1.0 - alpha)], dim=1)  # [B,3]
    else:
        main_w = torch.full((b, 1), alpha, device=device, dtype=dtype)
        other = (1.0 - alpha) / 2.0
        else_w = torch.full((b, 2), other, device=device, dtype=dtype)
        tea_w = torch.cat([main_w, else_w], dim=1)

    kd_loss = (tea_w * teacher_loss).sum(dim=1).mean() * float(kd_w)

    debug = {
        "kd_fuse": loss_fuse.mean().detach(),
        "kd_palm": loss_palm.mean().detach(),
        "kd_vein": loss_vein.mean().detach(),
        "alpha": torch.tensor(alpha, device=device),
    }
    return kd_loss, debug


def train_joint_distill(log_dir='runs_distill'):

    writer = SummaryWriter(log_dir=log_dir)
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher()

    cnn_palm_S, feat_dim_S, _ = build_backbone('tiny_mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('tiny_mobilefacenet')

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512,
        bottleneck=128, gate_hidden=32, final_l2norm=True).to(config.device)

    train_loader, val_loader, num_classes = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch)

    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=20.0,
        m=0.10,
    ).to(config.device)

    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.p2_epochs)

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    

    beta_kd = 0.85    # 融合特征蒸馏权重，可以根据效果调整

    # ---- MoVE-KD hyper-params (optional; will default if config lacks them) ----
    # MoVE-KD uses alpha=kd_memory_w to allocate a fixed weight to the main teacher,
    # and distributes the remaining (1-alpha) over other teachers.
    kd_w = float(getattr(config, 'kd_w', 1.0))
    kd_memory_w = float(getattr(config, 'kd_memory_w', 0.85))   # alpha in MoVE-KD
    use_token_weight = bool(getattr(config, 'token_weight', True))
    use_teacher_weight = bool(getattr(config, 'teacher_weight', True))
    teacher_temp = float(getattr(config, 'teacher_temp', 1.0))

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
            # ---------- 计算损失 ----------
            loss_ce = ce_loss(logits_S, labels)

            # ---------- MoVE-KD-style distillation (minimal adaptation) ----------
            kd_loss, kd_dbg = move_kd_loss_from_features(
                fused_S=fused_S, fused_T=fused_T,
                palm_S=F_palm_S, palm_T=F_palm_T,
                vein_S=F_vein_S, vein_T=F_vein_T,
                kd_w=kd_w,
                kd_memory_w=kd_memory_w,
                use_token_weight=use_token_weight,
                use_teacher_weight=use_teacher_weight,
                teacher_temp=teacher_temp,
            )

            loss = loss_ce + beta_kd * kd_loss

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

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'cnn_palm': cnn_palm_S.state_dict(),
                'cnn_vein': cnn_vein_S.state_dict(),
                'fusion': fusion_S.state_dict(),
                'classifier': classifier_S.state_dict(),
            }, os.path.join(config.save_dir, 'distill_best.pth'))

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
