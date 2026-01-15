import warnings
warnings.filterwarnings('ignore')

import os
from typing import Dict, List, Optional, Tuple

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
    """VkD-style projector with orthogonal parametrization when supported."""
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    if orthogonal is not None:
        lin = orthogonal(lin)
    return lin


def vkd_feature_loss(s: torch.Tensor,
                     t: torch.Tensor,
                     teacher_ln: bool = True,
                     normalize: bool = True) -> torch.Tensor:
    """VkD-like distillation:
    - task-specific normalization on teacher (LayerNorm)
    - (optional) L2-normalize then SmoothL1 for stable alignment
    """
    if teacher_ln:
        t = F.layer_norm(t, (t.shape[1],))
    if normalize:
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)
    return F.smooth_l1_loss(s, t)


def vkd_repr_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """VkD repr distill (teacher LN + SmoothL1)."""
    t = F.layer_norm(t, (t.shape[1],))
    return F.smooth_l1_loss(s, t)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy in percent."""
    with torch.no_grad():
        k = min(k, logits.size(1))
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1)).any(dim=1).float().mean().item()
        return 100.0 * correct


# ------------------------------
# Label remap helpers (避免 label 不连续/不一致导致 acc 恒为 0)
# ------------------------------
def parse_label_ids_from_txt(txt_path: str) -> List[int]:
    """Parse label ids from dataset txt file.
    Assumption: label id is the last whitespace-separated token of each non-empty line.
    """
    ids: List[int] = []
    if (txt_path is None) or (not os.path.exists(txt_path)):
        return ids
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ids.append(int(parts[-1]))
            except Exception:
                continue
    return ids


def build_label_mapping(train_txt: str, val_txt: str) -> Dict[int, int]:
    """Build mapping from raw label ids -> [0..C-1] using train+val txt files."""
    train_ids = parse_label_ids_from_txt(train_txt)
    val_ids = parse_label_ids_from_txt(val_txt)
    all_ids = sorted(set(train_ids + val_ids))
    if len(all_ids) == 0:
        return {}
    return {raw: i for i, raw in enumerate(all_ids)}


def remap_labels(labels: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    """Remap label tensor using mapping. If mapping is empty, return as-is."""
    if not mapping:
        return labels
    lbl = labels.detach().to("cpu").tolist()
    mapped = [mapping[int(x)] for x in lbl]
    return torch.tensor(mapped, device=labels.device, dtype=torch.long)


# ------------------------------
# ArcFace no-margin logits (关键！)
# 你的 train/val acc 一直 0，很可能是因为你用的是“带 margin 的 logits”
# 带 margin 的 logits 会刻意压低 GT 类别相似度，所以用它做 argmax 统计会极不合理，甚至全 0。
#
# 参考做法：像很多分类 repo 一样，
# 训练用 ArcFace logits（带 margin）算 loss，
# 但评估/统计精度用 “no-margin cosine logits”。
# ------------------------------
def _find_class_weight(head: nn.Module, num_classes: int, emb_dim: int) -> torch.Tensor:
    """
    尽量鲁棒地从 Arcface_Head 里找到 class weight:
    - 常见名字: weight / kernel / W
    - 也可能隐藏在 parameters 里，形状是 [C, D] 或 [D, C]
    """
    # common attribute names
    for name in ["weight", "kernel", "W"]:
        if hasattr(head, name):
            w = getattr(head, name)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                return w

    # fallback: search parameters by shape
    for _, p in head.named_parameters(recurse=True):
        if p.ndim != 2:
            continue
        if p.shape == (num_classes, emb_dim) or p.shape == (emb_dim, num_classes):
            return p

    # last resort: first 2D param
    for _, p in head.named_parameters(recurse=True):
        if p.ndim == 2:
            return p

    raise RuntimeError("Cannot find class weight matrix in Arcface_Head.")


def arcface_logits_no_margin(head: nn.Module,
                             emb: torch.Tensor,
                             num_classes: int,
                             emb_dim: int,
                             scale: Optional[float] = None) -> torch.Tensor:
    """
    计算不带 margin 的 logits：
      logits = s * (normalize(x) @ normalize(W)^T)
    用于：
    - 训练过程统计 acc（不要用带 margin 的 logits）
    - 验证/测试（推荐用 no-margin）
    """
    w = _find_class_weight(head, num_classes=num_classes, emb_dim=emb_dim)
    x = F.normalize(emb, dim=1)
    if w.shape[0] == num_classes:
        w_n = F.normalize(w, dim=1)
        logits = F.linear(x, w_n)  # [B, C]
    else:
        # shape [D, C]
        w_n = F.normalize(w, dim=0)
        logits = x @ w_n  # [B, C]

    s = scale
    if s is None:
        s = float(getattr(head, "s", 1.0))
    return logits * s


# ------------------------------
# Teacher builder
# ------------------------------
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

    for m in [cnn_palm_T, cnn_vein_T, fusion_T]:
        m.to(config.device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    print(f"==> Loaded teacher from {ckpt_path}")
    return cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T


def train_joint_distill(log_dir='runs_distill_vkd_opt_v3'):
    writer = SummaryWriter(log_dir=log_dir)

    # --------- teacher ----------
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher()

    # --------- student ----------
    cnn_palm_S, feat_dim_S, _ = build_backbone('tiny_mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('tiny_mobilefacenet')

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512,
        bottleneck=128, gate_hidden=32, final_l2norm=True
    ).to(config.device)

    train_loader, val_loader, num_classes_from_loader = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch
    )

    # --------- label remap ----------
    label_map = build_label_mapping(config.phase2_train, config.phase2_val)
    if label_map:
        num_classes = len(label_map)
        raw_ids = sorted(label_map.keys())
        print(f"[LabelMap] enabled: raw label range [{raw_ids[0]}..{raw_ids[-1]}], mapped classes={num_classes}")
    else:
        num_classes = num_classes_from_loader
        print(f"[LabelMap] disabled (fallback). num_classes(from loader)={num_classes}")

    # --------- ArcFace head ----------
    target_margin = 0.10
    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=20.0,
        m=target_margin,
    ).to(config.device)

    # --------- hooks for pre-proj repr distill ----------
    cache_t, cache_s = {}, {}

    def hook_t(module, inputs, output):
        cache_t['preproj'] = inputs[0]  # [B, 256]

    def hook_s(module, inputs, output):
        cache_s['preproj'] = inputs[0]  # [B, 128]

    h_t = fusion_T.proj.register_forward_hook(hook_t)
    h_s = fusion_S.proj.register_forward_hook(hook_s)

    # --------- projectors ----------
    proj_preproj = make_orth_linear(128, 256, bias=False).to(config.device)
    proj_palm = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)
    proj_vein = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)

    # --------- optimizer ----------
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    ce_loss = nn.CrossEntropyLoss()

    # --------- strategy ----------
    ce_only_epochs = max(3, int(0.05 * config.p2_epochs))
    warmup_epochs = max(10, int(0.15 * config.p2_epochs))

    beta_kd_max = 0.4
    alpha_repr_max = 0.2

    margin_warmup_epochs = max(5, int(0.05 * config.p2_epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0

    for epoch in range(config.p2_epochs):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        # margin warmup（训练算 loss 用带 margin 的 logits）
        if hasattr(classifier_S, 'm'):
            if epoch < margin_warmup_epochs:
                classifier_S.m = target_margin * float(epoch + 1) / float(margin_warmup_epochs)
            else:
                classifier_S.m = target_margin

        # distill weights
        if epoch < ce_only_epochs:
            beta_kd = 0.0
            alpha_repr = 0.0
        else:
            prog = min(1.0, float(epoch - ce_only_epochs + 1) / float(warmup_epochs))
            beta_kd = beta_kd_max * prog
            alpha_repr = alpha_repr_max * prog

        sum_loss = sum_xe = sum_kd = sum_repr = 0.0
        correct = total = 0
        top5_sum = 0.0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[VkD-OPT-v3] Ep {epoch+1}/{config.p2_epochs} beta={beta_kd:.3f} alpha={alpha_repr:.3f} m={getattr(classifier_S,"m",0):.3f}',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device, non_blocking=True)
            vein_img = vein_img.to(config.device, non_blocking=True)
            labels   = labels.to(config.device, non_blocking=True)
            labels = remap_labels(labels, label_map)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # teacher
                with torch.no_grad():
                    F_palm_T = cnn_palm_T(palm_img, return_spatial=False)
                    F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
                    fused_T  = fusion_T(F_palm_T, F_vein_T)
                    preproj_T = cache_t.get('preproj', None)

                # student
                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)
                preproj_S = cache_s.get('preproj', None)

                # ---- training logits (with margin) for loss ----
                logits_arc = classifier_S(fused_S, labels)
                loss_xe = ce_loss(logits_arc, labels)

                # ---- KD ----
                loss_kd = torch.tensor(0.0, device=config.device)
                if beta_kd > 0:
                    palm_S_kd = proj_palm(F_palm_S)
                    vein_S_kd = proj_vein(F_vein_S)
                    loss_kd = (
                        1.0 * vkd_feature_loss(fused_S, fused_T, teacher_ln=False, normalize=True) +
                        0.5 * (vkd_feature_loss(palm_S_kd, F_palm_T, teacher_ln=True, normalize=True) +
                               vkd_feature_loss(vein_S_kd, F_vein_T, teacher_ln=True, normalize=True))
                    )

                # ---- repr distill on fusion pre-proj ----
                loss_repr = torch.tensor(0.0, device=config.device)
                if alpha_repr > 0 and (preproj_T is not None) and (preproj_S is not None):
                    preproj_S_256 = proj_preproj(preproj_S)
                    loss_repr = vkd_repr_loss(preproj_S_256, preproj_T)

                loss = loss_xe + beta_kd * loss_kd + alpha_repr * loss_repr

            scaler.scale(loss).backward()
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

            sum_loss += float(loss.detach().cpu())
            sum_xe += float(loss_xe.detach().cpu())
            sum_kd += float(loss_kd.detach().cpu())
            sum_repr += float(loss_repr.detach().cpu())

            # ---- 训练精度统计：必须用 no-margin logits ----
            with torch.no_grad():
                logits_nom = arcface_logits_no_margin(classifier_S, fused_S, num_classes=num_classes, emb_dim=512)
                _, pred = torch.max(logits_nom, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                top5_sum += topk_accuracy(logits_nom, labels, k=5)

            pbar.update(1)

        pbar.close()

        tr_loss = sum_loss / len(train_loader)
        tr_xe = sum_xe / len(train_loader)
        tr_kd = sum_kd / len(train_loader)
        tr_repr = sum_repr / len(train_loader)
        tr_acc = 100.0 * correct / max(1, total)
        tr_acc5 = top5_sum / len(train_loader)

        # ---------------- Validation ----------------
        cnn_palm_S.eval()
        cnn_vein_S.eval()
        fusion_S.eval()
        classifier_S.eval()

        val_loss, val_correct, val_total = 0.0, 0, 0
        val_top5_sum = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device, non_blocking=True)
                vein_img = vein_img.to(config.device, non_blocking=True)
                labels   = labels.to(config.device, non_blocking=True)
                labels = remap_labels(labels, label_map)

                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)

                # 验证：统一用 no-margin logits 来算 loss/acc（更贴近推理）
                logits_nom = arcface_logits_no_margin(classifier_S, fused_S, num_classes=num_classes, emb_dim=512)
                loss = ce_loss(logits_nom, labels)
                val_loss += float(loss.detach().cpu())

                _, pred = torch.max(logits_nom, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                val_top5_sum += topk_accuracy(logits_nom, labels, k=5)

        va_loss = val_loss / len(val_loader)
        va_acc = 100.0 * val_correct / max(1, val_total)
        va_acc5 = val_top5_sum / len(val_loader)

        print(f"[VkD-OPT-v3][Epoch {epoch+1}] "
              f"TrLoss={tr_loss:.4f} (XE={tr_xe:.4f}, KD={tr_kd:.4f}, Repr={tr_repr:.4f}) "
              f"TrAcc={tr_acc:.4f}% TrAcc@5={tr_acc5:.4f}% | "
              f"VaLoss={va_loss:.4f}, VaAcc={va_acc:.4f}% VaAcc@5={va_acc5:.4f}%")

        writer.add_scalar('Train/TotalLoss', tr_loss, epoch)
        writer.add_scalar('Train/XE', tr_xe, epoch)
        writer.add_scalar('Train/KD', tr_kd, epoch)
        writer.add_scalar('Train/Repr', tr_repr, epoch)
        writer.add_scalar('Train/Acc1', tr_acc, epoch)
        writer.add_scalar('Train/Acc5', tr_acc5, epoch)

        writer.add_scalar('Val/Loss', va_loss, epoch)
        writer.add_scalar('Val/Acc1', va_acc, epoch)
        writer.add_scalar('Val/Acc5', va_acc5, epoch)

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
                'label_map': label_map,
            }, os.path.join(config.save_dir, 'distill_best.pth'))

        if early_stop(-va_acc, mode='min'):
            print(f"[VkD-OPT-v3] Early stopping at epoch {epoch+1}")
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
    print(f"[VkD-OPT-v3] Final best val acc: {best:.4f}%")
