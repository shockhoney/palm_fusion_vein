import warnings
warnings.filterwarnings('ignore')

import os
from typing import Dict, List, Optional

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
    # teacher task-specific normalization
    if teacher_ln:
        t = F.layer_norm(t, (t.shape[1],))
    # optional directional alignment
    if normalize:
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)
    return F.smooth_l1_loss(s, t)


def vkd_repr_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = F.layer_norm(t, (t.shape[1],))
    return F.smooth_l1_loss(s, t)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    with torch.no_grad():
        k = min(k, logits.size(1))
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1)).any(dim=1).float().mean().item()
        return 100.0 * correct


# ------------------------------
# Label remap helpers
# ------------------------------
def parse_label_ids_from_txt(txt_path: str) -> List[int]:
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
    train_ids = parse_label_ids_from_txt(train_txt)
    val_ids = parse_label_ids_from_txt(val_txt)
    all_ids = sorted(set(train_ids + val_ids))
    if len(all_ids) == 0:
        return {}
    return {raw: i for i, raw in enumerate(all_ids)}


def remap_labels(labels: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    if not mapping:
        return labels.long()
    lbl = labels.detach().to("cpu").tolist()
    mapped = [mapping[int(x)] for x in lbl]
    return torch.tensor(mapped, device=labels.device, dtype=torch.long)


# ------------------------------
# ArcFace no-margin logits (用于统计精度 & KL 蒸馏)
# ------------------------------
def _find_class_weight(head: nn.Module, num_classes: int, emb_dim: int) -> torch.Tensor:
    for name in ["weight", "kernel", "W"]:
        if hasattr(head, name):
            w = getattr(head, name)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                return w
    for _, p in head.named_parameters(recurse=True):
        if p.ndim == 2 and (p.shape == (num_classes, emb_dim) or p.shape == (emb_dim, num_classes)):
            return p
    for _, p in head.named_parameters(recurse=True):
        if p.ndim == 2:
            return p
    raise RuntimeError("Cannot find class weight matrix in Arcface_Head.")


def arcface_logits_no_margin(head: nn.Module,
                             emb: torch.Tensor,
                             num_classes: int,
                             emb_dim: int,
                             scale: Optional[float] = None) -> torch.Tensor:
    w = _find_class_weight(head, num_classes=num_classes, emb_dim=emb_dim)
    x = F.normalize(emb, dim=1)
    if w.shape[0] == num_classes:
        w_n = F.normalize(w, dim=1)
        logits = F.linear(x, w_n)  # [B, C]
    else:
        w_n = F.normalize(w, dim=0)
        logits = x @ w_n  # [B, C]
    s = scale if scale is not None else float(getattr(head, "s", 1.0))
    return logits * s



# ------------------------------
# 仅用 teacher embedding 构造“类别原型(prototype)”来做 KL 蒸馏（teacher 没有 classifier 权重时的替代方案）
# 思路：对每个类别，统计 teacher 的 fused embedding 均值作为该类原型，然后用 cosine similarity 得到 logits。
# 这样可以得到与分类任务一致的“soft targets”，并且不需要 teacher 保存 classifier。
# ------------------------------
@torch.no_grad()
def compute_teacher_prototypes(train_loader,
                               label_map: Dict[int, int],
                               num_classes: int,
                               cnn_palm_T,
                               cnn_vein_T,
                               fusion_T,
                               device,
                               emb_dim: int = 512) -> torch.Tensor:
    sums = torch.zeros(num_classes, emb_dim, dtype=torch.float32)
    cnts = torch.zeros(num_classes, dtype=torch.long)

    for palm_img, vein_img, labels in tqdm(train_loader, desc='[Proto] Build teacher prototypes', dynamic_ncols=True):
        palm_img = palm_img.to(device, non_blocking=True)
        vein_img = vein_img.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        labels = remap_labels(labels, label_map)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            F_palm_T = cnn_palm_T(palm_img, return_spatial=False)
            F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
            fused_T  = fusion_T(F_palm_T, F_vein_T)  # [B,512] (通常已 L2Norm)

        # 转 cpu 统计更稳，且内存小（500x512）
        emb = fused_T.detach().float().cpu()
        y = labels.detach().cpu()

        for i in range(emb.size(0)):
            cls = int(y[i].item())
            sums[cls] += emb[i]
            cnts[cls] += 1

    # 均值 + normalize
    cnts = torch.clamp(cnts, min=1).unsqueeze(1).float()
    proto = sums / cnts
    proto = F.normalize(proto, dim=1)  # [C,512]
    return proto.to(device)


def proto_logits(emb: torch.Tensor, proto: torch.Tensor, scale: float = 20.0) -> torch.Tensor:

    emb_n = F.normalize(emb, dim=1)
    return scale * F.linear(emb_n, proto)  # proto: [C,512] => acts like weight


# ------------------------------
# Build teacher (and optionally teacher classifier for KL distill)
# ------------------------------
def build_stage2_teacher(num_classes: int):

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



def train_joint_distill(log_dir='runs_distill_vkd_opt_v4_fast_proto_v2'):
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader, num_classes_from_loader = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch
    )

    # label mapping (you confirmed train/val both 500, still keep this for safety)
    label_map = build_label_mapping(config.phase2_train, config.phase2_val)
    if label_map:
        num_classes = len(label_map)
        raw_ids = sorted(label_map.keys())
        print(f"[LabelMap] enabled: raw label range [{raw_ids[0]}..{raw_ids[-1]}], mapped classes={num_classes}")
    else:
        num_classes = num_classes_from_loader
        print(f"[LabelMap] disabled. num_classes(from loader)={num_classes}")

    # --------- teacher ----------
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher(num_classes)

    # ---------- v4-fast-proto：teacher 没有 classifier 权重时，用 teacher embedding 构造类别原型做 KL 蒸馏 ----------
    # 这一步只需要跑一遍 train_loader（开销很小），得到 [C,512] 的 proto 矩阵。
    teacher_proto = compute_teacher_prototypes(
        train_loader=train_loader,
        label_map=label_map,
        num_classes=num_classes,
        cnn_palm_T=cnn_palm_T,
        cnn_vein_T=cnn_vein_T,
        fusion_T=fusion_T,
        device=config.device,
        emb_dim=512
    )
    print("==> Built teacher prototypes for KL distillation.")

    # --------- student ----------
    cnn_palm_S, feat_dim_S, _ = build_backbone('tiny_mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('tiny_mobilefacenet')

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512,
        bottleneck=128, gate_hidden=32, final_l2norm=True
    ).to(config.device)

    # ArcFace student head (train uses margin)
    target_margin = 0.10
    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=20.0,
        m=target_margin,
    ).to(config.device)

    # hooks for pre-proj repr distill
    cache_t, cache_s = {}, {}

    def hook_t(module, inputs, output):
        cache_t['preproj'] = inputs[0]  # [B, 256]

    def hook_s(module, inputs, output):
        cache_s['preproj'] = inputs[0]  # [B, 128]

    h_t = fusion_T.proj.register_forward_hook(hook_t)
    h_s = fusion_S.proj.register_forward_hook(hook_s)

    # projectors
    proj_preproj = make_orth_linear(128, 256, bias=False).to(config.device)
    proj_palm = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)
    proj_vein = make_orth_linear(feat_dim_S, feat_dim_T, bias=False).to(config.device)

    # optimizer: slightly increase head lr to learn faster
    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr * 4.0},
        {'params': proj_preproj.parameters(),'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
        {'params': proj_palm.parameters(),  'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
        {'params': proj_vein.parameters(),  'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    ce_loss = nn.CrossEntropyLoss()  
    kl = nn.KLDivLoss(reduction='batchmean')

    # Strategy:
    # - longer CE-only to get stronger baseline
    ce_only_epochs = 3  # v4-fast: 更早进入蒸馏
    warmup_epochs = 10  # v4-fast: 蒸馏权重更快warmup

    # weights (make KL the main distill for classification, like VkD's gamma)
    beta_feat_max = 0.2        # feature KD (small)
    alpha_repr_max = 0.15      # repr distill (small)
    gamma_kl_max = 2.0         # v2: 提高KL权重（teacher无classifier时更依赖原型KL）

    T = 1.5                    # v2: 温度更低，KL信号更强

    proto_scale = 64.0          # v2: 原型logits放大(类似ArcFace的s)，让soft targets更尖锐

    # margin warmup longer (ArcFace is harder for small student)
    margin_warmup_epochs = 60  # v4-fast: 更久时间保持小margin，XE下降更快

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0

    for epoch in range(config.p2_epochs):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        # margin warmup
        if hasattr(classifier_S, 'm'):
            if epoch < margin_warmup_epochs:
                classifier_S.m = target_margin * float(epoch + 1) / float(margin_warmup_epochs)
            else:
                classifier_S.m = target_margin

        # weights schedule
        if epoch < ce_only_epochs:
            beta_feat = 0.0
            alpha_repr = 0.0
            gamma_kl = 0.0
        else:
            prog = min(1.0, float(epoch - ce_only_epochs + 1) / float(warmup_epochs))
            beta_feat = beta_feat_max * prog
            alpha_repr = alpha_repr_max * prog
            gamma_kl = gamma_kl_max * prog

        sum_loss = sum_xe = sum_feat = sum_repr = sum_kl = 0.0
        correct = total = 0
        top5_sum = 0.0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[VkD-OPT-v4-PROTO] Ep {epoch+1}/{config.p2_epochs} '
                         f'beta={beta_feat:.3f} alpha={alpha_repr:.3f} gamma={gamma_kl:.3f} m={getattr(classifier_S,"m",0):.3f}',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device, non_blocking=True)
            vein_img = vein_img.to(config.device, non_blocking=True)
            labels   = labels.to(config.device, non_blocking=True)
            labels = remap_labels(labels, label_map)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # teacher forward
                with torch.no_grad():
                    F_palm_T = cnn_palm_T(palm_img, return_spatial=False)
                    F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
                    fused_T  = fusion_T(F_palm_T, F_vein_T)
                    preproj_T = cache_t.get('preproj', None)

                    # teacher logits for KL distill (用 prototype，不需要 teacher classifier)
                    logits_T = None
                    if gamma_kl > 0:
                        logits_T = proto_logits(fused_T, teacher_proto, scale=proto_scale)

                # student forward
                F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
                F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
                fused_S  = fusion_S(F_palm_S, F_vein_S)
                preproj_S = cache_s.get('preproj', None)

                # train logits (with margin) for XE
                logits_arc = classifier_S(fused_S, labels)
                loss_xe = ce_loss(logits_arc, labels)

                # feature KD
                loss_feat = torch.tensor(0.0, device=config.device)
                if beta_feat > 0:
                    palm_S_kd = proj_palm(F_palm_S)
                    vein_S_kd = proj_vein(F_vein_S)
                    loss_feat = (
                        1.0 * vkd_feature_loss(fused_S, fused_T, teacher_ln=False, normalize=True) +
                        0.5 * (vkd_feature_loss(palm_S_kd, F_palm_T, teacher_ln=True, normalize=True) +
                               vkd_feature_loss(vein_S_kd, F_vein_T, teacher_ln=True, normalize=True))
                    )

                # repr distill (pre-proj)
                loss_repr = torch.tensor(0.0, device=config.device)
                if alpha_repr > 0 and (preproj_T is not None) and (preproj_S is not None):
                    preproj_S_256 = proj_preproj(preproj_S)
                    loss_repr = vkd_repr_loss(preproj_S_256, preproj_T)

                # KL distill on logits (VkD-style gamma)
                loss_kl = torch.tensor(0.0, device=config.device)
                if gamma_kl > 0 and logits_T is not None:
                    logits_S_nom = proto_logits(fused_S, teacher_proto, scale=proto_scale)  # 用同一套原型做 student logits
                    loss_kl = kl(F.log_softmax(logits_S_nom / T, dim=-1),
                                 F.softmax(logits_T / T, dim=-1)) * (T * T)

                loss = loss_xe + beta_feat * loss_feat + alpha_repr * loss_repr + gamma_kl * loss_kl

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
            sum_feat += float(loss_feat.detach().cpu())
            sum_repr += float(loss_repr.detach().cpu())
            sum_kl += float(loss_kl.detach().cpu())

            # stats acc with no-margin logits
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
        tr_feat = sum_feat / len(train_loader)
        tr_repr = sum_repr / len(train_loader)
        tr_kl = sum_kl / len(train_loader)
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

        print(f"[VkD-OPT-v4-PROTO][Epoch {epoch+1}] "
              f"TrLoss={tr_loss:.4f} (XE={tr_xe:.4f}, Feat={tr_feat:.4f}, Repr={tr_repr:.4f}, KL={tr_kl:.6f}) "
              f"TrAcc={tr_acc:.4f}% TrAcc@5={tr_acc5:.4f}% | "
              f"VaLoss={va_loss:.4f}, VaAcc={va_acc:.4f}% VaAcc@5={va_acc5:.4f}%")

        # TB logs
        writer.add_scalar('Train/TotalLoss', tr_loss, epoch)
        writer.add_scalar('Train/XE', tr_xe, epoch)
        writer.add_scalar('Train/FeatKD', tr_feat, epoch)
        writer.add_scalar('Train/Repr', tr_repr, epoch)
        writer.add_scalar('Train/KL', tr_kl, epoch)
        writer.add_scalar('Train/Acc1', tr_acc, epoch)
        writer.add_scalar('Train/Acc5', tr_acc5, epoch)

        writer.add_scalar('Val/Loss', va_loss, epoch)
        writer.add_scalar('Val/Acc1', va_acc, epoch)
        writer.add_scalar('Val/Acc5', va_acc5, epoch)

        writer.add_scalar('Sched/beta_feat', beta_feat, epoch)
        writer.add_scalar('Sched/alpha_repr', alpha_repr, epoch)
        writer.add_scalar('Sched/gamma_kl', gamma_kl, epoch)
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
            print(f"[VkD-OPT-v4-PROTO] Early stopping at epoch {epoch+1}")
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
    print(f"[VkD-OPT-v4-PROTO] Final best val acc: {best:.4f}%")
