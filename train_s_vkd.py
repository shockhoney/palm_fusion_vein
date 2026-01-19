import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets_txt import PairTxtDataset
from utils.metrics import compute_eer, tar_at_far
from utils.head import Arcface_Head

from models.stage1_mobileFacenet import MobileFaceNet
from models.student_mobilefacenet import TinyMobileFaceNet
from models.stage2 import Stage2Fusion
from models.student_fusion import Stage2FusionStudent_BottleneckGate


# -------------------------
# utils: pair scores + metrics
# -------------------------
def build_pair_scores(feats: np.ndarray, labels: np.ndarray):
    feats = feats.astype(np.float32)
    labels = labels.astype(np.int64)
    sim = feats @ feats.T  # feats already L2-normalized => cosine
    n = labels.shape[0]
    i, j = np.triu_indices(n, k=1)
    scores = sim[i, j].astype(np.float32)
    pair_labels = (labels[i] == labels[j]).astype(np.int32)
    return scores, pair_labels


def get_tar_value(tar_ret):
    # tar_at_far 在不同实现中可能返回 float 或 dict
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


@torch.no_grad()
def evaluate_eer_tar(cnn_palm, cnn_vein, fusion, loader, device, far_list):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion.eval()

    feats, labs = [], []
    for palm, vein, y in loader:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)

        fp = cnn_palm(palm, return_spatial=False)
        fv = cnn_vein(vein, return_spatial=False)
        z = fusion(fp, fv)
        z = F.normalize(z, dim=1)

        feats.append(z.cpu().numpy())
        labs.append(y.numpy())

    feats = np.vstack(feats)
    labs = np.concatenate(labs, axis=0)

    scores, pair_labels = build_pair_scores(feats, labs)

    # 按你项目 test.py 的常见签名：compute_eer(scores, pair_labels, is_similarity=True, ...)
    eer = compute_eer(scores, pair_labels, is_similarity=True)

    tar_msg = []
    for far in far_list:
        tar_ret = tar_at_far(scores, pair_labels, far, is_similarity=True)
        tar_val = get_tar_value(tar_ret)
        tar_msg.append((far, tar_val))

    return float(eer), tar_msg


# -------------------------
# KD losses (concise)
# -------------------------
def cosine_kd(z_s, z_t):
    z_s = F.normalize(z_s, dim=1)
    z_t = F.normalize(z_t, dim=1)
    # per-sample (B,)
    return 1.0 - (z_s * z_t).sum(dim=1)


def sim_matrix(z):
    z = F.normalize(z, dim=1)
    return z @ z.T


def weighted_relational_kd(z_s, z_t, w):
    """
    Relational KD: align similarity matrices in a batch.
    w: (B,) teacher-confidence weights in [0,1]
    """
    S_s = sim_matrix(z_s)
    S_t = sim_matrix(z_t)
    W = (w[:, None] * w[None, :]).detach()
    return ((S_s - S_t) ** 2 * W).mean()


def main():
    parser = argparse.ArgumentParser("Distill student (TinyMobileFaceNet) from teacher (MobileFaceNet)")
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

    # distill weights
    parser.add_argument("--lambda_emb", type=float, default=2.0)
    parser.add_argument("--lambda_rel", type=float, default=2.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)

    # ramp-up (MoVE-KD style: gradually emphasize KD)
    parser.add_argument("--ramp_epochs", type=int, default=30)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # num_classes from train_list
    # -------------------------
    label_set = set()
    with open(args.train_list, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                label_set.add(int(p[2]))
    num_classes = len(label_set)
    print(f"[Info] num_classes = {num_classes}")

    # -------------------------
    # transforms (match teacher training style)
    # -------------------------
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_set = PairTxtDataset(args.train_list, transform_palm=tf_train, transform_vein=tf_train)
    val_set = PairTxtDataset(args.val_list, transform_palm=tf_val, transform_vein=tf_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # -------------------------
    # Teacher: MobileFaceNet + Stage2Fusion + classifier
    # -------------------------
    cnn_palm_T = MobileFaceNet(input_channel=3, input_size=224).to(device)
    cnn_vein_T = MobileFaceNet(input_channel=3, input_size=224).to(device)
    feat_dim_T = cnn_palm_T.out_dim

    fusion_T = Stage2Fusion(in_dim_global=feat_dim_T, out_dim_final=512, final_l2norm=True).to(device)
    classifier_T = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)

    cnn_palm_T.load_state_dict(ckpt["cnn_palm"], strict=True)
    cnn_vein_T.load_state_dict(ckpt["cnn_vein"], strict=True)
    fusion_T.load_state_dict(ckpt["fusion"], strict=True)
    classifier_T.load_state_dict(ckpt["classifier"], strict=True)

    for m in [cnn_palm_T, cnn_vein_T, fusion_T, classifier_T]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # -------------------------
    # Student: TinyMobileFaceNet + StudentFusion + classifier
    # -------------------------
    cnn_palm_S = TinyMobileFaceNet(input_channel=3, embedding_size=256).to(device)
    cnn_vein_S = TinyMobileFaceNet(input_channel=3, embedding_size=256).to(device)
    feat_dim_S = cnn_palm_S.out_dim

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512, final_l2norm=True
    ).to(device)
    classifier_S = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    optimizer = torch.optim.AdamW(
        list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) + list(classifier_S.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    ce = nn.CrossEntropyLoss()

    def ramp_w(epoch):
        if args.ramp_epochs <= 0:
            return 1.0
        return min(1.0, epoch / float(args.ramp_epochs))

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_distill.pth")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        w = ramp_w(epoch)
        lam_emb = args.lambda_emb * w
        lam_rel = args.lambda_rel * w

        total_loss, total_n = 0.0, 0

        for palm, vein, y in train_loader:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                fp_T = cnn_palm_T(palm, return_spatial=False)
                fv_T = cnn_vein_T(vein, return_spatial=False)
                z_T = fusion_T(fp_T, fv_T)  # (B,512)
                # teacher confidence (MoVE-KD spirit: emphasize valuable samples)
                logit_T = classifier_T(z_T, y)
                conf = F.softmax(logit_T, dim=1).max(dim=1).values.clamp(0.0, 1.0)  # (B,)

            # ---- student forward ----
            fp_S = cnn_palm_S(palm, return_spatial=False)
            fv_S = cnn_vein_S(vein, return_spatial=False)
            z_S = fusion_S(fp_S, fv_S)  # (B,512)

            logit_S = classifier_S(z_S, y)
            loss_cls = ce(logit_S, y)

            # embedding KD (weighted)
            emb_per = cosine_kd(z_S, z_T)                 # (B,)
            loss_emb = (emb_per * conf).sum() / (conf.sum() + 1e-6)

            # relational KD (weighted)
            loss_rel = weighted_relational_kd(z_S, z_T, conf)

            loss = args.lambda_cls * loss_cls + lam_emb * loss_emb + lam_rel * loss_rel

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) + list(fusion_S.parameters()),
                1.0
            )
            optimizer.step()

            bs = palm.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        print(f"Epoch [{epoch}/{args.epochs}] loss={total_loss/max(total_n,1):.4f} "
              f"| kd_w={w:.2f} (emb={lam_emb:.2f}, rel={lam_rel:.2f})")

        # ---- eval every N epochs ----
        if epoch % args.eval_every == 0:
            eer, tar_list = evaluate_eer_tar(cnn_palm_S, cnn_vein_S, fusion_S, val_loader, device, args.far_list)
            tar_str = " ".join([f"TAR@FAR={far:.0e}:{tar*100:.2f}%" for far, tar in tar_list])
            print(f"[VAL] Epoch {epoch}: EER={eer*100:.2f}% | {tar_str}")

            if eer < best_eer:
                best_eer = eer
                torch.save({
                    "cnn_palm": cnn_palm_S.state_dict(),
                    "cnn_vein": cnn_vein_S.state_dict(),
                    "fusion": fusion_S.state_dict(),
                    "classifier": classifier_S.state_dict(),
                }, best_path)
                print(f"[SAVE] best_eer={best_eer*100:.2f}% -> {best_path}")

    last_path = os.path.join(args.save_dir, "student_last_distill.pth")
    torch.save({
        "cnn_palm": cnn_palm_S.state_dict(),
        "cnn_vein": cnn_vein_S.state_dict(),
        "fusion": fusion_S.state_dict(),
        "classifier": classifier_S.state_dict(),
    }, last_path)
    print(f"[DONE] last={last_path}, best={best_path}")


if __name__ == "__main__":
    main()
