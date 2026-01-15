import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.student_fusion import Stage2FusionStudent_BottleneckGate    
from train_teacher import config, build_backbone, create_phase2_dataloaders
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



# ------------------------------
# Verification metrics (TAR / EER)
# 训练过程中不再使用分类准确率，只在验证阶段计算 TAR/EER 等验证指标
# ------------------------------
def _roc_from_pair_scores(scores: torch.Tensor, is_genuine: torch.Tensor):
    """
    根据 pairwise score 和 genuine mask 计算 ROC 曲线点。
    scores: (P,) 越大越相似
    is_genuine: (P,) bool
    返回: fpr, tpr, thresholds (均为 1D tensor, 已按阈值从高到低排列)
    """
    scores = scores.float().view(-1)
    is_genuine = is_genuine.bool().view(-1)
    order = torch.argsort(scores, descending=True)
    scores_sorted = scores[order]
    genuine_sorted = is_genuine[order]

    n_pos = genuine_sorted.sum().clamp_min(1).float()
    n_neg = (~genuine_sorted).sum().clamp_min(1).float()

    tp = torch.cumsum(genuine_sorted.to(torch.int64), dim=0).float()
    fp = torch.cumsum((~genuine_sorted).to(torch.int64), dim=0).float()
    tpr = tp / n_pos
    fpr = fp / n_neg
    return fpr, tpr, scores_sorted


def compute_eer_and_tar(embeddings: torch.Tensor, labels: torch.Tensor, far_list=(1e-4, 1e-3, 1e-2)):
    """
    计算 EER 和 TAR@FAR。
    embeddings: (N, D) 特征
    labels: (N,) 类别/身份
    """
    emb = F.normalize(embeddings, p=2, dim=1)
    sim = emb @ emb.t()  # cosine similarity, (N,N)
    n = sim.size(0)
    iu = torch.triu_indices(n, n, offset=1)
    scores = sim[iu[0], iu[1]].cpu()
    is_genuine = (labels[iu[0]] == labels[iu[1]]).cpu()

    fpr, tpr, _ = _roc_from_pair_scores(scores, is_genuine)
    frr = 1.0 - tpr
    diff = torch.abs(fpr - frr)
    k = torch.argmin(diff)
    eer = float((fpr[k] + frr[k]) / 2.0)

    tar_at_far = {}
    for far in far_list:
        far = float(far)
        mask = fpr <= far
        tar_at_far[far] = float(tpr[mask][-1]) if mask.any() else 0.0
    return eer, tar_at_far



def collect_embeddings_and_loss(model_parts, classifier, loader, ce_loss, device, max_samples=None):
    """Collect normalized fused embeddings and labels for verification metrics.
    This uses the same forward path as validation/testing.
    To keep runtime reasonable on large train sets, you can cap max_samples.
    """
    cnn_palm_S, cnn_vein_S, fusion_S = model_parts
    cnn_palm_S.eval(); cnn_vein_S.eval(); fusion_S.eval(); classifier.eval()

    embs = []
    labs = []
    total_loss = 0.0
    num_batches = 0
    num_seen = 0

    with torch.no_grad():
        for palm_img, vein_img, labels in loader:
            palm_img = palm_img.to(device)
            vein_img = vein_img.to(device)
            labels = labels.to(device)

            F_palm_S = cnn_palm_S(palm_img, return_spatial=False)
            F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
            fused_S  = fusion_S(F_palm_S, F_vein_S)

            logits_S = classifier(fused_S, labels)
            loss = ce_loss(logits_S, labels)
            total_loss += float(loss.item())
            num_batches += 1

            fused_S_n = F.normalize(fused_S, p=2, dim=1)
            embs.append(fused_S_n.detach().cpu())
            labs.append(labels.detach().cpu())

            num_seen += labels.size(0)
            if max_samples is not None and num_seen >= int(max_samples):
                break

    if num_batches == 0:
        return torch.empty(0, 512), torch.empty(0, dtype=torch.long), 0.0

    embs = torch.cat(embs, dim=0)
    labs = torch.cat(labs, dim=0)
    if max_samples is not None and embs.size(0) > int(max_samples):
        embs = embs[: int(max_samples)]
        labs = labs[: int(max_samples)]
    avg_loss = total_loss / num_batches
    return embs, labs, avg_loss

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

    beta_kd = 0.85    # 融合特征蒸馏权重，可以根据效果调整

    # 以验证集 EER 最低作为最佳模型（越低越好）
    best_eer = 1.0
    best_tar_1e3 = 0.0

    for epoch in range(config.p2_epochs):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        train_loss = 0.0

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

            # ---------- Student 前向 ----------
            F_palm_S = cnn_palm_S(palm_img, return_spatial=False)      # [B, feat_dim_S]
            F_vein_S = cnn_vein_S(vein_img, return_spatial=False)
            fused_S  = fusion_S(F_palm_S, F_vein_S)                    # [B, 512]

            logits_S = classifier_S(fused_S, labels)

            # ---------- 计算损失（只保留 loss，不再计算/打印分类准确率） ----------
            loss_ce = ce_loss(logits_S, labels)
            loss_kd_fuse = kd_cosine(fused_S, fused_T)
            loss_kd_palm = kd_cosine(F_palm_S, F_palm_T)
            loss_kd_vein = kd_cosine(F_vein_S, F_vein_T)

            loss_kd = 1.0 * loss_kd_fuse + 0.5 * (loss_kd_palm + loss_kd_vein)
            loss = loss_ce + beta_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_palm_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_S.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier_S.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.update(1)

        pbar.close()
        avg_train_loss = train_loss / len(train_loader)

        # ---------- Train verification metrics (EER/TAR) ----------
        # Note: full train set pairwise scoring is O(N^2). To keep runtime reasonable,
        # we cap the number of samples used for train EER/TAR via config.train_eval_samples.
        train_eval_samples = int(getattr(config, 'train_eval_samples', 2000))
        train_embs, train_labs, _ = collect_embeddings_and_loss(
            model_parts=(cnn_palm_S, cnn_vein_S, fusion_S),
            classifier=classifier_S,
            loader=train_loader,
            ce_loss=ce_loss,
            device=config.device,
            max_samples=train_eval_samples if train_eval_samples > 0 else None,
        )
        if train_embs.numel() > 0:
            train_eer, train_tar = compute_eer_and_tar(train_embs, train_labs, far_list=(1e-4, 1e-3, 1e-2))
            train_tar_1e3 = train_tar.get(1e-3, 0.0)
        else:
            train_eer, train_tar_1e3 = 1.0, 0.0



        # ---------- 验证：loss + (TAR/EER) ----------
        cnn_palm_S.eval()
        cnn_vein_S.eval()
        fusion_S.eval()
        classifier_S.eval()

        val_total_loss = 0.0
        val_embs = []
        val_labs = []

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

                fused_S_n = F.normalize(fused_S, p=2, dim=1)
                val_embs.append(fused_S_n.detach().cpu())
                val_labs.append(labels.detach().cpu())

        avg_val_loss = val_total_loss / len(val_loader)
        val_embs = torch.cat(val_embs, dim=0)
        val_labs = torch.cat(val_labs, dim=0)
        val_eer, val_tar = compute_eer_and_tar(val_embs, val_labs, far_list=(1e-4, 1e-3, 1e-2))
        val_tar_1e3 = val_tar.get(1e-3, 0.0)

        print(f"[JointDistill][Epoch {epoch+1}] TrLoss={avg_train_loss:.4f}, VaLoss={avg_val_loss:.4f}, " \
              f"TrainEER={train_eer*100:.2f}%, TrainTAR@FAR1e-3={train_tar_1e3*100:.2f}%, " \
              f"ValEER={val_eer*100:.2f}%, ValTAR@FAR1e-3={val_tar_1e3*100:.2f}%")

        # TensorBoard：只记录 loss + 验证指标
        writer.add_scalar('JointDistill/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('JointDistill/TrainEER',  train_eer,       epoch)
        writer.add_scalar('JointDistill/TrainTAR_FAR1e-3', train_tar_1e3, epoch)
        writer.add_scalar('JointDistill/ValLoss',   avg_val_loss,   epoch)
        writer.add_scalar('JointDistill/ValEER',    val_eer,        epoch)
        writer.add_scalar('JointDistill/TAR_FAR1e-4', val_tar.get(1e-4, 0.0), epoch)
        writer.add_scalar('JointDistill/TAR_FAR1e-3', val_tar.get(1e-3, 0.0), epoch)
        writer.add_scalar('JointDistill/TAR_FAR1e-2', val_tar.get(1e-2, 0.0), epoch)

        # 保存最优（EER 最低；如相等则 TAR@1e-3 更高）
        if (val_eer < best_eer) or (abs(val_eer - best_eer) < 1e-8 and val_tar_1e3 > best_tar_1e3):
            best_eer = val_eer
            best_tar_1e3 = val_tar_1e3
            torch.save({
                'cnn_palm': cnn_palm_S.state_dict(),
                'cnn_vein': cnn_vein_S.state_dict(),
                'fusion': fusion_S.state_dict(),
                'classifier': classifier_S.state_dict(),
            }, os.path.join(config.save_dir, 'distill_best.pth'))

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    return best_eer

if __name__ == '__main__':
    os.makedirs(config.save_dir, exist_ok=True)
    best = train_joint_distill()
    print(f"[JointDistill] Final best EER: {best*100:.2f}%")