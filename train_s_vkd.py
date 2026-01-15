import warnings
warnings.filterwarnings('ignore')

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.student_fusion import Stage2FusionStudent_BottleneckGate
from train_teacher import config, build_backbone, create_phase2_dataloaders, EarlyStopping
from models.stage2 import Stage2Fusion
from utils.head import Arcface_Head

# ==============================
# VkD: orthogonal projector (auto fallback)
# ==============================
try:
    from torch.nn.utils.parametrizations import orthogonal
except Exception:
    orthogonal = None


def make_orth_linear(in_dim: int, out_dim: int, bias: bool = False) -> nn.Module:
    """VkD-style projector.
    - Prefer orthogonal parametrization when available.
    - Fallback: orthogonal init only (still stable, but not strictly constrained).
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    if orthogonal is not None:
        lin = orthogonal(lin)
    return lin


def vkd_repr_loss(student_feat: torch.Tensor,
                  teacher_feat: torch.Tensor,
                  layernorm_teacher: bool = True) -> torch.Tensor:
    """Representation distillation used in VkD classification code:
    teacher-side task norm (LayerNorm), SmoothL1 for robustness.
    """
    if layernorm_teacher:
        teacher_feat = F.layer_norm(teacher_feat, (teacher_feat.shape[1],))
    return F.smooth_l1_loss(student_feat, teacher_feat)


def vkd_feature_loss(s: torch.Tensor,
                     t: torch.Tensor,
                     teacher_ln: bool = True,
                     normalize: bool = True) -> torch.Tensor:
    """Feature distillation helper:
    - optional LayerNorm on teacher (task-specific normalization)
    - optional L2 normalization before SmoothL1 (directional alignment, stable)
    """
    if teacher_ln:
        t = F.layer_norm(t, (t.shape[1],))
    if normalize:
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)
    return F.smooth_l1_loss(s, t)


def build_stage2_teacher():
    cnn_palm_T, feat_dim_T, _ = build_backbone('mobilefacenet')
    cnn_vein_T, _, _          = build_backbone('mobilefacenet')
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

    # freeze teacher
    for m in [cnn_palm_T, cnn_vein_T, fusion_T]:
        m.to(config.device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    print(f"==> Loaded teacher from {ckpt_path}")
    return cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T


def train_joint_distill(log_dir='runs_distill_vkd_opt'):
    writer = SummaryWriter(log_dir=log_dir)

    # ---------------- Teacher & Student ----------------
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher()

    cnn_palm_S, feat_dim_S, _ = build_backbone('tiny_mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('tiny_mobilefacenet')

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512,
        bottleneck=128, gate_hidden=32, final_l2norm=True
    ).to(config.device)

    train_loader, val_loader, num_classes = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch
    )

    # ArcFace head (train uses margin; eval will temporarily set m=0)
    target_margin = 0.10
    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=20.0,
        m=target_margin,
    ).to(config.device)

    # ---------------- VkD: capture pre-proj representations via hooks ----------------
    # teacher: fusion_T.proj input  -> 256-d fused representation before proj
    # student: fusion_S.proj input  -> 128-d bottleneck fused representation before proj
    cache_t, cache_s = {}, {}

    def hook_t(module, inputs, output):
        cache_t['preproj'] = inputs[0]

    def hook_s(module, inputs, output):
        cache_s['preproj'] = inputs[0]

    h_t = fusion_T.proj.register_forward_hook(hook_t)
    h_s = fusion_S.proj.register_forward_hook(hook_s)

    # ---------------- VkD projectors ----------------
    # representation distillation projector: 128 -> 256
    proj_preproj = make_orth_linear(128, 256, bias=False).to(config.device)
    # feature distillation projectors (align dims, optional but usually stabilizes KD)
    proj_palm = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)
    proj_vein = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)

    # ---------------- Optimizer & sched ----------------
    # 关键：让 projector 的 lr 更小更稳（类似 VkD 中 projector 是辅助项）
    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr},
        {'params': proj_preproj.parameters(),'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
        {'params': proj_palm.parameters(),  'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
        {'params': proj_vein.parameters(),  'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)

    # per-epoch cosine (保持你原有风格，最小改动)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    ce_loss = nn.CrossEntropyLoss()

    # ---------------- Training strategy (关键优化点) ----------------
    # 1) 先让 student 用 CE “站稳”几轮（否则蒸馏很容易把随机网络拉偏）
    ce_only_epochs = max(3, int(0.05 * config.p2_epochs))

    # 2) 然后再 warmup 蒸馏权重（参考 VkD：alpha/gamma 都是可控开关）
    warmup_epochs = max(10, int(0.15 * config.p2_epochs))

    beta_kd_max = 0.4      # 原来 0.85 太激进，容易压死 CE（尤其你 batch 小、类多）
    alpha_repr_max = 0.2   # pre-proj repr 蒸馏更强，权重要更保守

    # 3) ArcFace margin 也 warmup：前期 m=0，更像普通分类，稳定后再恢复到 target_margin
    margin_warmup_epochs = max(5, int(0.05 * config.p2_epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0

    for epoch in range(config.p2_epochs):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        # --- margin schedule ---
        if hasattr(classifier_S, 'm'):
            if epoch < margin_warmup_epochs:
                classifier_S.m = target_margin * float(epoch + 1) / float(margin_warmup_epochs)
            else:
                classifier_S.m = target_margin

        # --- distill weight schedule ---
        if epoch < ce_only_epochs:
            beta_kd = 0.0
            alpha_repr = 0.0
        else:
            # linearly warm up
            prog = min(1.0, float(epoch - ce_only_epochs + 1) / float(warmup_epochs))
            beta_kd = beta_kd_max * prog
            alpha_repr = alpha_repr_max * prog

        # meters
        sum_loss = 0.0
        sum_xe = 0.0
        sum_kd = 0.0
        sum_repr = 0.0
        correct = 0
        total = 0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[VkD-OPT] Epoch {epoch+1}/{config.p2_epochs} (beta={beta_kd:.3f}, alpha={alpha_repr:.3f}, m={getattr(classifier_S,"m",0):.3f})',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device, non_blocking=True)
            vein_img = vein_img.to(config.device, non_blocking=True)
            labels   = labels.to(config.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # --- teacher forward (no grad) ---
                with torch.no_grad():
                    F_palm_T = cnn_palm_T(palm_img, return_spatial=False)
                    F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
                    fused_T  = fusion_T(F_palm_T, F_vein_T)
                    preproj_T = cache_t.get('preproj', None)

                # --- student forward ---
                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)
                preproj_S = cache_s.get('preproj', None)

                # --- XE (task loss) ---
                logits_S = classifier_S(fused_S, labels)
                loss_xe = ce_loss(logits_S, labels)

                # --- KD losses (VkD-style: projector + teacher normalization + smoothl1) ---
                loss_kd = torch.tensor(0.0, device=config.device)
                loss_repr = torch.tensor(0.0, device=config.device)

                if beta_kd > 0:
                    palm_S_kd = proj_palm(F_palm_S)
                    vein_S_kd = proj_vein(F_vein_S)
                    # feature distill (方向 + 稳健损失)
                    loss_kd = (
                        1.0 * vkd_feature_loss(fused_S, fused_T, teacher_ln=False, normalize=True) +
                        0.5 * (vkd_feature_loss(palm_S_kd, F_palm_T, teacher_ln=True, normalize=True) +
                               vkd_feature_loss(vein_S_kd, F_vein_T, teacher_ln=True, normalize=True))
                    )

                if alpha_repr > 0 and (preproj_T is not None) and (preproj_S is not None):
                    preproj_S_256 = proj_preproj(preproj_S)
                    loss_repr = vkd_repr_loss(preproj_S_256, preproj_T, layernorm_teacher=True)

                loss = loss_xe + beta_kd * loss_kd + alpha_repr * loss_repr

            scaler.scale(loss).backward()
            # grad clip (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cnn_palm_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_preproj.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_palm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_vein.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # metrics
            sum_loss += float(loss.detach().cpu())
            sum_xe += float(loss_xe.detach().cpu())
            sum_kd += float(loss_kd.detach().cpu())
            sum_repr += float(loss_repr.detach().cpu())

            _, pred = torch.max(logits_S, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.update(1)

        pbar.close()

        tr_loss = sum_loss / len(train_loader)
        tr_xe = sum_xe / len(train_loader)
        tr_kd = sum_kd / len(train_loader)
        tr_repr = sum_repr / len(train_loader)
        tr_acc = 100.0 * correct / max(1, total)

        # ---------------- Validation ----------------
        cnn_palm_S.eval()
        cnn_vein_S.eval()
        fusion_S.eval()
        classifier_S.eval()

        # eval: ArcFace 不加 margin（更接近推理）
        old_m = getattr(classifier_S, 'm', None)
        if old_m is not None:
            classifier_S.m = 0.0

        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device, non_blocking=True)
                vein_img = vein_img.to(config.device, non_blocking=True)
                labels   = labels.to(config.device, non_blocking=True)

                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)

                logits_S = classifier_S(fused_S, labels)
                loss = ce_loss(logits_S, labels)
                val_loss += float(loss.detach().cpu())

                _, pred = torch.max(logits_S, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        if old_m is not None:
            classifier_S.m = old_m

        va_loss = val_loss / len(val_loader)
        va_acc = 100.0 * val_correct / max(1, val_total)

        print(f"[VkD-OPT][Epoch {epoch+1}] "
              f"TrLoss={tr_loss:.4f} (XE={tr_xe:.4f}, KD={tr_kd:.4f}, Repr={tr_repr:.4f}) "
              f"TrAcc={tr_acc:.4f}% | VaLoss={va_loss:.4f}, VaAcc={va_acc:.4f}%")

        # TB logs
        writer.add_scalar('Train/TotalLoss', tr_loss, epoch)
        writer.add_scalar('Train/XE', tr_xe, epoch)
        writer.add_scalar('Train/KD', tr_kd, epoch)
        writer.add_scalar('Train/Repr', tr_repr, epoch)
        writer.add_scalar('Train/Acc', tr_acc, epoch)

        writer.add_scalar('Val/Loss', va_loss, epoch)
        writer.add_scalar('Val/Acc', va_acc, epoch)

        writer.add_scalar('Sched/beta_kd', beta_kd, epoch)
        writer.add_scalar('Sched/alpha_repr', alpha_repr, epoch)
        writer.add_scalar('Sched/margin_m', getattr(classifier_S, 'm', 0.0), epoch)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                'cnn_palm': cnn_palm_S.state_dict(),
                'cnn_vein': cnn_vein_S.state_dict(),
                'fusion': fusion_S.state_dict(),
                'classifier': classifier_S.state_dict(),
                'proj_preproj': proj_preproj.state_dict(),
                'proj_palm': proj_palm.state_dict(),
                'proj_vein': proj_vein.state_dict(),
            }, os.path.join(config.save_dir, 'distill_best.pth'))

        if early_stop(-va_acc, mode='min'):
            print(f"[VkD-OPT] Early stopping at epoch {epoch+1}")
            break

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    h_t.remove(); h_s.remove()
    return best_acc


if __name__ == '__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    best = train_joint_distill()
    print(f"[VkD-OPT] Final best val acc: {best:.4f}%")
