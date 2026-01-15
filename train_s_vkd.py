import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.student_fusion import Stage2FusionStudent_BottleneckGate    
from utils.kd_loss import total_loss
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


def make_orth_linear(in_dim: int, out_dim: int, bias: bool = False) -> nn.Module:
    """VkD-style orthogonal projector.
    If torch supports orthogonal parametrization, constrain W to be (semi-)orthogonal;
    otherwise keep orthogonal initialization (best-effort).
    """
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    if orthogonal is not None:
        lin = orthogonal(lin)
    return lin


def vkd_feature_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    use_layernorm: bool = True,
    match_norm: bool = False,
    norm_weight: float = 0.05,
) -> torch.Tensor:
    """VkD-style feature distillation:
    - optional LayerNorm (task-specific normalization)
    - align directions with SmoothL1 on L2-normalized features
    - optional norm matching (usually OFF if you already LayerNorm)
    """
    if use_layernorm:
        s = F.layer_norm(s, (s.size(1),))
        t = F.layer_norm(t, (t.size(1),))
    s_n = F.normalize(s, dim=1)
    t_n = F.normalize(t, dim=1)
    loss = F.smooth_l1_loss(s_n, t_n)
    if match_norm:
        loss = loss + norm_weight * F.smooth_l1_loss(s.norm(dim=1), t.norm(dim=1))
    return loss


def kd_cosine(s, t):
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


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


    # ---------- VkD (Plan B): distill fusion pre-proj representations ----------
    # Capture teacher's 256-d fused_feat BEFORE fusion_T.proj, and student's 128-d bottleneck BEFORE fusion_S.proj
    _hook_cache = {'t_preproj': None, 's_bottleneck': None}

    def _hook_t_proj(module, inp, out):
        # inp[0]: [B, 256] (pre-proj feature)
        _hook_cache['t_preproj'] = inp[0].detach()

    def _hook_s_proj(module, inp, out):
        # inp[0]: [B, bottleneck=128] (student bottleneck feature, keep graph)
        _hook_cache['s_bottleneck'] = inp[0]

    # Register hooks once
    _t_handle = fusion_T.proj.register_forward_hook(_hook_t_proj)
    _s_handle = fusion_S.proj.register_forward_hook(_hook_s_proj)

    # Project student bottleneck (128) -> teacher pre-proj space (256) with (semi-)orthogonal projector
    proj_preproj_kd = make_orth_linear(128, feat_dim_T, bias=False).to(config.device)

    # Weight for pre-proj distillation (keep small at first; tune if needed)
    w_preproj = 0.5

    params = [
        {'params': cnn_palm_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein_S.parameters(), 'lr': config.p2_enc_lr},
        {'params': fusion_S.parameters(),   'lr': config.p2_lr},
        {'params': classifier_S.parameters(),'lr': config.p2_lr},
        {'params': proj_preproj_kd.parameters(), 'lr': config.p2_lr},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.p2_epochs)

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    

    beta_kd = 0.80    # 融合特征蒸馏权重，可以根据效果调整

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
            loss_kd_fuse = kd_cosine(fused_S, fused_T)
            loss_kd_palm = kd_cosine(F_palm_S, F_palm_T)
            loss_kd_vein = kd_cosine(F_vein_S, F_vein_T)

            # Pre-proj distillation: student bottleneck (128) -> teacher pre-proj (256)
            # (hook values are set during fusion_T(...) and fusion_S(...) forward above)
            if _hook_cache['t_preproj'] is None or _hook_cache['s_bottleneck'] is None:
                raise RuntimeError("Pre-proj hook cache missing. Check that fusion_T.proj and fusion_S.proj exist.")
            s_pre = proj_preproj_kd(_hook_cache['s_bottleneck'])   # [B, 256]
            t_pre = _hook_cache['t_preproj']                       # [B, 256]
            loss_kd_preproj = vkd_feature_loss(s_pre, t_pre, use_layernorm=True, match_norm=False)

            loss_kd = 1.0 * loss_kd_fuse + 0.5 * (loss_kd_palm + loss_kd_vein) + w_preproj * loss_kd_preproj
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

    # remove hooks
    try:
        _t_handle.remove()
        _s_handle.remove()
    except Exception:
        pass

    writer.close()
    return best_acc


if __name__ == '__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    best = train_joint_distill()
    print(f"[JointDistill] Final best val acc: {best:.2f}%")