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

# ==============================
# VkD-style orthogonal projector（如果 PyTorch 版本较旧则自动降级）
# ==============================
try:
    from torch.nn.utils.parametrizations import orthogonal
except Exception:
    orthogonal = None


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

    # 冻结 teacher
    for m in [cnn_palm_T, cnn_vein_T, fusion_T]:
        m.to(config.device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    print(f"==> Loaded teacher from {ckpt_path}")
    return cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T


def kd_cosine(s, t):
    """原始 KD：对齐方向（余弦）。"""
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def make_orth_linear(in_dim: int, out_dim: int, bias: bool = False) -> nn.Module:
    """
    VkD 核心点：用（半）正交映射作为 projector，避免普通 Linear 学出缩放/剪切，导致蒸馏信号“扭曲”。
    - 如果环境支持 torch.nn.utils.parametrizations.orthogonal，就会强制保持正交参数化。
    - 否则退化为“正交初始化的 Linear”（不会报错，但不再严格保持正交）。
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    if orthogonal is not None:
        lin = orthogonal(lin)
    return lin


def freeze_bn_stats(m: nn.Module):
    """
    小 batch + MobileFaceNet(BN很多) 时，BN running stats 容易漂移，
    导致验证精度“缓慢下降”。这里冻结 BN 的 running_mean/var（不冻结权重）。
    如果你 batch 足够大或已用 SyncBN，也可以注释掉调用。
    """
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def train_joint_distill(log_dir='runs_distill'):
    writer = SummaryWriter(log_dir=log_dir)

    # ========== Teacher ==========
    cnn_palm_T, cnn_vein_T, fusion_T, feat_dim_T = build_stage2_teacher()

    # ========== Student ==========
    cnn_palm_S, feat_dim_S, _ = build_backbone('tiny_mobilefacenet')
    cnn_vein_S, _, _          = build_backbone('tiny_mobilefacenet')

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512,
        bottleneck=128, gate_hidden=32, final_l2norm=True
    ).to(config.device)

    train_loader, val_loader, num_classes = create_phase2_dataloaders(
        config.phase2_train, config.phase2_val, config.p2_batch
    )

    classifier_S = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=20.0,
        m=0.10,
    ).to(config.device)

    # ==============================
    # 方案B（最小改动实现）：
    # 用 forward hook 抓 fusion 的 “proj 之前” 表征：
    # - teacher: self.proj 输入是 256 维 fused_feat（proj 之前）
    # - student: self.proj 输入是 bottleneck 128 维 fused（proj 之前）
    # 然后新增 projector: 128 -> 256，对齐后用 SmoothL1 蒸馏（参考 VkD 的 repr distill）
    # ==============================
    cache_t = {}
    cache_s = {}

    def hook_t(module, inputs, output):
        # inputs 是 tuple，inputs[0] 形状 [B, 256]
        cache_t['preproj'] = inputs[0]

    def hook_s(module, inputs, output):
        # inputs[0] 形状 [B, 128]
        cache_s['preproj'] = inputs[0]

    # 只 hook 到 proj 层（不改模型 forward / 不影响加载 ckpt）
    h_t = fusion_T.proj.register_forward_hook(hook_t)
    h_s = fusion_S.proj.register_forward_hook(hook_s)

    # VkD projector：student bottleneck(128) -> teacher preproj(256)
    proj_preproj = make_orth_linear(128, 256, bias=False).to(config.device)

    # 训练参数
    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr},
        # projector 额外加进 optimizer（lr 可以更小更稳）
        {'params': proj_preproj.parameters(),'lr': config.p2_lr * 0.1, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    ce_loss = nn.CrossEntropyLoss()

    # ======= 蒸馏权重（参考 VkD 思路：蒸馏别一上来就很强，先 warmup）=======
    beta_kd_max = 0.85            # 你原来的 KD 权重（cosine KD）
    alpha_repr_max = 0.5          # pre-proj 表征蒸馏权重（新加的 repr distill）
    warmup_epochs = max(1, int(0.1 * config.p2_epochs))

    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0

    # ======= 训练 =======
    for epoch in range(config.p2_epochs):
        # warmup：前 warmup_epochs 逐步增大蒸馏权重
        warm = min(1.0, float(epoch + 1) / float(warmup_epochs))
        beta_kd = beta_kd_max * warm
        alpha_repr = alpha_repr_max * warm

        cnn_palm_S.train(); cnn_palm_S.apply(freeze_bn_stats)
        cnn_vein_S.train(); cnn_vein_S.apply(freeze_bn_stats)
        fusion_S.train();   fusion_S.apply(freeze_bn_stats)
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
                F_palm_T = cnn_palm_T(palm_img, return_spatial=False)  # [B, feat_dim_T]
                F_vein_T = cnn_vein_T(vein_img, return_spatial=False)
                fused_T  = fusion_T(F_palm_T, F_vein_T)               # [B, 512]
                # hook_t 会把 fusion_T.proj 之前的 256 维表征放入 cache_t['preproj']
                preproj_T = cache_t.get('preproj', None)

            # ---------- Student 前向 ----------
            F_palm_S = cnn_palm_S(palm_img, return_spatial=False)     # [B, feat_dim_S]
            F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
            fused_S  = fusion_S(F_palm_S, F_vein_S)                   # [B, 512]
            # hook_s 会把 fusion_S.proj 之前的 128 维表征放入 cache_s['preproj']
            preproj_S = cache_s.get('preproj', None)

            # ---------- 任务损失 ----------
            logits_S = classifier_S(fused_S, labels)
            loss_ce = ce_loss(logits_S, labels)

            # ---------- 原始 KD（cosine） ----------
            loss_kd_fuse = kd_cosine(fused_S, fused_T)
            loss_kd_palm = kd_cosine(F_palm_S, F_palm_T)
            loss_kd_vein = kd_cosine(F_vein_S, F_vein_T)
            loss_kd = 1.0 * loss_kd_fuse + 0.5 * (loss_kd_palm + loss_kd_vein)

            # ---------- 方案B：pre-proj 表征蒸馏（参考 VkD 的 repr distill） ----------
            # 说明：
            # 1) 先用正交 projector 把 student bottleneck(128) 映射到 teacher preproj(256)
            # 2) 对 teacher 的 preproj 做 layer_norm（参考 VkD 代码：z_t_conv_norm = layer_norm(...)）
            # 3) 用 SmoothL1 做对齐（鲁棒，梯度更稳定）
            repr_loss = torch.tensor(0.0, device=config.device)
            if (preproj_T is not None) and (preproj_S is not None):
                preproj_S_256 = proj_preproj(preproj_S)
                preproj_T_norm = F.layer_norm(preproj_T, (preproj_T.shape[1],))
                repr_loss = F.smooth_l1_loss(preproj_S_256, preproj_T_norm)

            # 总损失（参考 VkD：loss = xe + kl + repr；这里 kl 不好做就不加）
            loss = loss_ce + beta_kd * loss_kd + alpha_repr * repr_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_palm_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_preproj.parameters(), 1.0)

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

        # 重要：ArcFace 在验证时一般不加 margin，否则“看起来精度下降”
        # 这里做一个最小侵入的处理：临时把 m 置 0，算完再恢复
        old_m = getattr(classifier_S, 'm', None)
        if old_m is not None:
            classifier_S.m = 0.0

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

                vloss = ce_loss(logits_S, labels)
                val_total_loss += vloss.item()
                _, pred = torch.max(logits_S, 1)
                val_correct += (pred == labels).sum().item()
                val_total   += labels.size(0)

        if old_m is not None:
            classifier_S.m = old_m

        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_acc  = 100. * val_correct / val_total

        print(f"[JointDistill][Epoch {epoch+1}] "
              f"TrLoss={avg_train_loss:.4f}, TrAcc={avg_train_acc:.2f}%, "
              f"VaLoss={avg_val_loss:.4f}, VaAcc={avg_val_acc:.2f}%")

        # 写 tensorboard（多记录 repr_loss 和权重，方便排查）
        writer.add_scalar('JointDistill/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('JointDistill/TrainAcc',  avg_train_acc,  epoch)
        writer.add_scalar('JointDistill/ValLoss',   avg_val_loss,   epoch)
        writer.add_scalar('JointDistill/ValAcc',    avg_val_acc,    epoch)
        writer.add_scalar('JointDistill/beta_kd',   beta_kd,        epoch)
        writer.add_scalar('JointDistill/alpha_repr',alpha_repr,     epoch)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'cnn_palm': cnn_palm_S.state_dict(),
                'cnn_vein': cnn_vein_S.state_dict(),
                'fusion': fusion_S.state_dict(),
                'classifier': classifier_S.state_dict(),
                'proj_preproj': proj_preproj.state_dict(),
            }, os.path.join(config.save_dir, 'distill_best.pth'))

        if early_stop(-avg_val_acc, mode='min'):
            print(f"[JointDistill] Early stopping at epoch {epoch+1}")
            break

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()

    # 清理 hook（避免重复注册）
    h_t.remove()
    h_s.remove()

    return best_acc


if __name__ == '__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    best = train_joint_distill()
    print(f"[JointDistill] Final best val acc: {best:.2f}%")
