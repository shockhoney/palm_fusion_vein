import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.datasets_txt import PairTxtDataset
from utils.metrics import compute_eer, tar_at_far
from utils.head import Arcface_Head

from models.stage1_mobileFacenet import MobileFaceNet
from models.student_mobilefacenet import TinyMobileFaceNet
from models.stage2 import Stage2Fusion
from models.student_fusion import Stage2FusionStudent_BottleneckGate


# -------------------------
# pair scores + metrics
# -------------------------
def build_pair_scores(feats: np.ndarray, labels: np.ndarray):
    feats = feats.astype(np.float32)
    labels = labels.astype(np.int64)
    sim = feats @ feats.T
    n = labels.shape[0]
    i, j = np.triu_indices(n, k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def get_tar_value(tar_ret):
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


@torch.no_grad()
def evaluate_eer_tar(cnn_palm, cnn_vein, fusion, loader, device, far_list, desc="Val"):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion.eval()

    feats, labs = [], []
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for palm, vein, y in pbar:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)

        fp = backbone_global(cnn_palm, palm)  # always (B,D)
        fv = backbone_global(cnn_vein, vein)  # always (B,D)
        z = fusion(fp, fv)
        z = F.normalize(z, dim=1)

        feats.append(z.cpu().numpy())
        labs.append(y.numpy())

    feats = np.vstack(feats)
    labs = np.concatenate(labs, axis=0)
    scores, pair_labels = build_pair_scores(feats, labs)

    pos = int((pair_labels == 1).sum())
    neg = int((pair_labels == 0).sum())
    if pos == 0 or neg == 0:
        msg = (
            f"[WARN] Validation pairs invalid for EER/TAR: pos_pairs={pos}, neg_pairs={neg}. "
            f"通常是 val_list 中每个身份只出现 1 次导致没有 genuine pair。\n"
            f"建议：确保验证集每个身份≥2张，或使用 test.py 的 pair-protocol 评估。"
        )
        return float("nan"), [(far, float("nan")) for far in far_list], msg

    eer = compute_eer(scores, pair_labels, is_similarity=True)
    tar_list = []
    for far in far_list:
        tar_ret = tar_at_far(scores, pair_labels, far, is_similarity=True)
        tar_list.append((far, get_tar_value(tar_ret)))

    return float(eer), tar_list, ""


# -------------------------
# backbone output adapters (关键修复：确保 fusion 输入永远是 (B,D))
# -------------------------
def _to_global(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert feature to (B,D):
      - (B,D) -> (B,D)
      - (B,C,H,W) -> GAP -> (B,C)
      - (B,N,D) -> mean token -> (B,D)
    """
    if feat.ndim == 2:
        return feat
    if feat.ndim == 4:
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)
    if feat.ndim == 3:
        return feat.mean(dim=1)
    raise ValueError(f"Unsupported feature shape for global: {tuple(feat.shape)}")


def _pick_first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x):
                return x
    return None


def backbone_global(model, x):
    """
    Robustly get global embedding (B,D) from backbone.
    """
    out = None
    try:
        out = model(x, return_spatial=False)
    except TypeError:
        out = model(x)

    t = _pick_first_tensor(out)
    if t is None:
        raise ValueError("Backbone forward returned no tensor.")
    return _to_global(t)


def backbone_spatial(model, x):
    """
    Try to get spatial map (B,C,H,W) for token KD.
    - If model(x, return_spatial=True) returns (global, spatial) -> pick spatial
    - If returns a tensor and it's 4D -> use it
    Otherwise return None.
    """
    try:
        out = model(x, return_spatial=True)
    except TypeError:
        return None

    if torch.is_tensor(out) and out.ndim == 4:
        return out

    if isinstance(out, (tuple, list)):
        # prefer 4D tensor
        for item in out:
            if torch.is_tensor(item) and item.ndim == 4:
                return item

    return None


# -------------------------
# MoVE-KD style: token-weighted MSE (single teacher -> W_tea=1)
# -------------------------
def token_weight_from_teacher(t_tokens):
    # t_tokens: (B,N,D)
    score = t_tokens.pow(2).mean(dim=-1)  # (B,N)
    return F.softmax(score, dim=1)        # (B,N)


def token_weighted_mse(t_tokens, s_tokens, add_uniform=True):
    """
    L = sum_j (w_j + 1/N) * MSE(t_j, s_j)
    """
    b, n, _ = t_tokens.shape
    w = token_weight_from_teacher(t_tokens).detach()
    if add_uniform:
        w = w + (1.0 / float(n))
    mse_tok = (t_tokens - s_tokens).pow(2).mean(dim=-1)  # (B,N)
    return (w * mse_tok).sum(dim=1).mean()


def flatten_tokens(feat_4d, target_hw=None):
    """
    (B,C,H,W) -> (B,N,C)
    """
    if target_hw is not None and feat_4d.shape[-2:] != target_hw:
        feat_4d = F.interpolate(feat_4d, size=target_hw, mode="bilinear", align_corners=False)
    b, c, h, w = feat_4d.shape
    return feat_4d.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)


class TokenProjector(nn.Module):
    """
    Minimal adapter:
      - spatial: LazyConv2d -> kd_dim
      - fusion vector: Linear(512 -> kd_dim)
    """
    def __init__(self, kd_dim: int):
        super().__init__()
        self.proj2d = nn.LazyConv2d(kd_dim, kernel_size=1, bias=False)
        self.proj1d = nn.Linear(512, kd_dim, bias=False)

    def proj_spatial(self, feat_4d):
        return self.proj2d(feat_4d)

    def proj_fusion(self, vec_2d):
        return self.proj1d(vec_2d)


def ramp(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(ramp_epochs))


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser("MoVE-KD distillation for palm+vein fusion")
    parser.add_argument("--train_list", type=str, default="txt-datasets/polyu_phase2_train.txt")
    parser.add_argument("--val_list", type=str, default="txt-datasets/polyu_phase2_val.txt")
    parser.add_argument("--teacher_ckpt", type=str, default="outputs/models/stage2_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs/models")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    parser.add_argument("--kd_w", type=float, default=1.0)
    parser.add_argument("--kd_dim", type=int, default=128)
    parser.add_argument("--add_uniform", action="store_true")
    parser.add_argument("--ramp_epochs", type=int, default=30)

    parser.add_argument("--lambda_cls", type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_classes
    label_set = set()
    with open(args.train_list, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                label_set.add(int(p[2]))
    num_classes = len(label_set)
    print(f"[Info] num_classes = {num_classes}")

    # transforms
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set = PairTxtDataset(args.train_list, transform_palm=tf_train, transform_vein=tf_train)
    val_set = PairTxtDataset(args.val_list, transform_palm=tf_val, transform_vein=tf_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Teacher (frozen): MobileFaceNet + fusion + classifier
    cnn_palm_T = MobileFaceNet(input_channel=3, input_size=224).to(device)
    cnn_vein_T = MobileFaceNet(input_channel=3, input_size=224).to(device)
    feat_dim_T = cnn_palm_T.out_dim
    fusion_T = Stage2Fusion(in_dim_global=feat_dim_T, out_dim_final=512, final_l2norm=True).to(device)
    classifier_T = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    ckpt = safe_torch_load(args.teacher_ckpt, device)
    cnn_palm_T.load_state_dict(ckpt["cnn_palm"], strict=True)
    cnn_vein_T.load_state_dict(ckpt["cnn_vein"], strict=True)
    fusion_T.load_state_dict(ckpt["fusion"], strict=True)
    classifier_T.load_state_dict(ckpt["classifier"], strict=True)

    for m in [cnn_palm_T, cnn_vein_T, fusion_T, classifier_T]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Student
    cnn_palm_S = TinyMobileFaceNet(input_channel=3, embedding_size=256).to(device)
    cnn_vein_S = TinyMobileFaceNet(input_channel=3, embedding_size=256).to(device)
    feat_dim_S = cnn_palm_S.out_dim
    fusion_S = Stage2FusionStudent_BottleneckGate(in_dim_global=feat_dim_S, out_dim_final=512, final_l2norm=True).to(device)
    classifier_S = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    # MoVE-KD adapters
    proj_T = TokenProjector(args.kd_dim).to(device)
    proj_S = TokenProjector(args.kd_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) + list(classifier_S.parameters()) +
        list(proj_T.parameters()) + list(proj_S.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    ce = nn.CrossEntropyLoss()

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_move_kd.pth")

    warned_spatial = False

    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train(); cnn_vein_S.train(); fusion_S.train(); classifier_S.train()
        proj_T.train(); proj_S.train()

        kd_scale = args.kd_w * ramp(epoch, args.ramp_epochs)

        total_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for palm, vein, y in pbar:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # teacher
            with torch.no_grad():
                fp_T_g = backbone_global(cnn_palm_T, palm)  # (B,D) guaranteed
                fv_T_g = backbone_global(cnn_vein_T, vein)
                z_T = fusion_T(fp_T_g, fv_T_g)              # now safe: (B,512)

                fp_T_s = backbone_spatial(cnn_palm_T, palm)  # (B,C,H,W) or None
                fv_T_s = backbone_spatial(cnn_vein_T, vein)

            # student
            fp_S_g = backbone_global(cnn_palm_S, palm)
            fv_S_g = backbone_global(cnn_vein_S, vein)
            z_S = fusion_S(fp_S_g, fv_S_g)

            # cls
            logits_S = classifier_S(z_S, y)
            loss_cls = ce(logits_S, y)

            # MoVE-KD: token-weighted MSE
            kd_parts = []

            # fusion vector KD (simple MSE in KD space)
            t_f = proj_T.proj_fusion(z_T)
            s_f = proj_S.proj_fusion(z_S)
            kd_parts.append(F.mse_loss(s_f, t_f))

            # palm token KD
            fp_S_s = backbone_spatial(cnn_palm_S, palm)
            if fp_T_s is not None and fp_S_s is not None:
                t_map = proj_T.proj_spatial(fp_T_s)
                s_map = proj_S.proj_spatial(fp_S_s)
                t_tok, hw = flatten_tokens(t_map)
                s_tok, _ = flatten_tokens(s_map, target_hw=hw)
                kd_parts.append(token_weighted_mse(t_tok, s_tok, add_uniform=args.add_uniform))
            elif not warned_spatial:
                warned_spatial = True
                print("[WARN] backbone_spatial() 返回 None，说明你的 backbone 不支持 return_spatial=True 输出特征图。"
                      "此时将只做 fusion 向量 KD（仍可训练，但不完全等价 MoVE-KD token KD）。")

            # vein token KD
            fv_S_s = backbone_spatial(cnn_vein_S, vein)
            if fv_T_s is not None and fv_S_s is not None:
                t_map = proj_T.proj_spatial(fv_T_s)
                s_map = proj_S.proj_spatial(fv_S_s)
                t_tok, hw = flatten_tokens(t_map)
                s_tok, _ = flatten_tokens(s_map, target_hw=hw)
                kd_parts.append(token_weighted_mse(t_tok, s_tok, add_uniform=args.add_uniform))

            loss_kd = sum(kd_parts) / float(len(kd_parts))
            loss = args.lambda_cls * loss_cls + kd_scale * loss_kd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) + list(fusion_S.parameters()),
                1.0
            )
            optimizer.step()

            bs = palm.size(0)
            total_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{loss_cls.item():.4f}", kd=f"{loss_kd.item():.4f}", kd_w=f"{kd_scale:.2f}")

        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={total_loss/max(seen,1):.4f} | kd_w={kd_scale:.2f}")

        if epoch % args.eval_every == 0:
            eer, tar_list, warn = evaluate_eer_tar(
                cnn_palm_S, cnn_vein_S, fusion_S, val_loader, device, args.far_list, desc=f"Val@{epoch}"
            )
            if warn:
                print(warn)
                print(f"[VAL] Epoch {epoch}: EER=NaN | " +
                      " ".join([f"TAR@FAR={far:.0e}:NaN" for far, _ in tar_list]))
            else:
                tar_str = " ".join([f"TAR@FAR={far:.0e}:{tar*100:.2f}%" for far, tar in tar_list])
                print(f"[VAL] Epoch {epoch}: EER={eer*100:.2f}% | {tar_str}")

                if eer < best_eer:
                    best_eer = eer
                    torch.save({
                        "cnn_palm": cnn_palm_S.state_dict(),
                        "cnn_vein": cnn_vein_S.state_dict(),
                        "fusion": fusion_S.state_dict(),
                        "classifier": classifier_S.state_dict(),
                        "proj_T": proj_T.state_dict(),
                        "proj_S": proj_S.state_dict(),
                        "epoch": epoch,
                        "best_eer": best_eer
                    }, best_path)
                    print(f"[SAVE] best_eer={best_eer*100:.2f}% -> {best_path}")

    last_path = os.path.join(args.save_dir, "student_last_move_kd.pth")
    torch.save({
        "cnn_palm": cnn_palm_S.state_dict(),
        "cnn_vein": cnn_vein_S.state_dict(),
        "fusion": fusion_S.state_dict(),
        "classifier": classifier_S.state_dict(),
        "proj_T": proj_T.state_dict(),
        "proj_S": proj_S.state_dict(),
        "epoch": args.epochs,
        "best_eer": best_eer
    }, last_path)
    print(f"[DONE] last={last_path}, best={best_path}")


if __name__ == "__main__":
    main()
