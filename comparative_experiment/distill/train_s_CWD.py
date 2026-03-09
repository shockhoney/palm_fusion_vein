
import os
import sys

# Add the project root directory to sys.path so 'utils' and 'models' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
# pair scores
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
    # tar_at_far: some impl returns float, some returns dict
    if isinstance(tar_ret, dict):
        return float(tar_ret.get("TAR", tar_ret.get("tar", 0.0)))
    return float(tar_ret)


@torch.no_grad()
def evaluate_eer_tar(cnn_palm, cnn_vein, fusion, loader, device, far_list,
                    classifier=None, ce_fn=None, desc="Val"):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion.eval()

    feats, labs = [], []
    total_loss, correct, seen = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for palm, vein, y in pbar:
        palm = palm.to(device, non_blocking=True)
        vein = vein.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)

        fp = cnn_palm(palm, return_spatial=False)
        fv = cnn_vein(vein, return_spatial=False)
        z = fusion(fp, fv)
        z = F.normalize(z, dim=1)

        # loss & acc
        if classifier is not None and ce_fn is not None:
            logit = classifier(z, y_dev)
            total_loss += ce_fn(logit, y_dev).item() * palm.size(0)
            correct += (logit.argmax(1) == y_dev).sum().item()

        feats.append(z.cpu().numpy())
        labs.append(y.numpy())
        seen += palm.size(0)

    feats = np.vstack(feats)
    labs = np.concatenate(labs, axis=0)
    val_loss = total_loss / max(seen, 1)
    val_acc = correct / max(seen, 1)

    scores, pair_labels = build_pair_scores(feats, labs)

    pos = int((pair_labels == 1).sum())
    neg = int((pair_labels == 0).sum())
    if pos == 0 or neg == 0:
        msg = (
            f"[WARN] Validation pairs invalid for EER/TAR: pos_pairs={pos}, neg_pairs={neg}. "
            f"原因通常是 val_list 中每个身份只出现 1 次（无法形成 genuine pair），或协议不是“按身份两两配对”。\n"
            f"建议：确保验证集每个身份至少2张，或使用 test.py 的 pair-protocol（同/不同对）文件进行评估。"
        )
        return float("nan"), [(far, float("nan")) for far in far_list], msg, val_loss, val_acc

    eer = compute_eer(scores, pair_labels, is_similarity=True)
    tar_list = []
    for far in far_list:
        tar_ret = tar_at_far(scores, pair_labels, far, is_similarity=True)
        tar_list.append((far, get_tar_value(tar_ret)))

    return float(eer), tar_list, "", val_loss, val_acc


# -------------------------
# CWD loss
# -------------------------
class ChannelWiseDistillationLoss(nn.Module):
    """
    Channel-wise KD for dense prediction:
    For each channel, perform spatial softmax over HxW and minimize KL(teacher || student).
    """

    def __init__(self, temperature: float = 4.0, loss_weight: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        if student_feat.dim() != 4 or teacher_feat.dim() != 4:
            raise ValueError(
                f"CWD expects 4D tensors [B,C,H,W], got {student_feat.shape} and {teacher_feat.shape}"
            )
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                f"CWD requires same shape for student/teacher features, got "
                f"{student_feat.shape} vs {teacher_feat.shape}"
            )

        b, c, h, w = student_feat.shape
        t = self.temperature

        s = student_feat.view(b, c, -1) / t
        tea = teacher_feat.view(b, c, -1) / t

        s_log_prob = F.log_softmax(s, dim=-1)
        t_prob = F.softmax(tea, dim=-1)

        loss = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1).mean()
        return self.loss_weight * (t ** 2) * loss


def safe_torch_load(path, device):
    # Avoid torch.load warning in newer PyTorch if possible
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser("Train student with CWD distillation from teacher")
    parser.add_argument("--train_list", type=str, default="data_txt/polyu_phase2_train.txt")
    parser.add_argument("--val_list", type=str, default="data_txt/polyu_phase2_val.txt")
    parser.add_argument("--teacher_ckpt", type=str, default="outputs/polyu_models/stage2_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs_distill/CWD_models")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    # loss weights
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_cwd", type=float, default=1.0)
    parser.add_argument("--cwd_temperature", type=float, default=4.0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs_cwd"))

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
    # transforms (match teacher style)
    # -------------------------
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

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # -------------------------
    # Teacher: MobileFaceNet + Stage2Fusion + classifier (all in ckpt)
    # -------------------------
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

    cwd_loss = ChannelWiseDistillationLoss(
        temperature=args.cwd_temperature,
        loss_weight=1.0
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(cnn_palm_S.parameters()) +
        list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) +
        list(classifier_S.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    ce = nn.CrossEntropyLoss()

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_cwd.pth")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_cwd = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for palm, vein, y in pbar:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                feat_palm_T = cnn_palm_T(palm, return_spatial=True)   # [B,256,H,W]
                feat_vein_T = cnn_vein_T(vein, return_spatial=True)   # [B,256,H,W]
                fp_T = cnn_palm_T(palm, return_spatial=False)         # [B,256]
                fv_T = cnn_vein_T(vein, return_spatial=False)         # [B,256]
                z_T = fusion_T(fp_T, fv_T)                            # [B,512]
                _ = classifier_T(z_T, y)                              # keep teacher path complete/consistent

            # ---- student forward ----
            feat_palm_S = cnn_palm_S(palm, return_spatial=True)       # [B,256,H,W]
            feat_vein_S = cnn_vein_S(vein, return_spatial=True)       # [B,256,H,W]
            fp_S = cnn_palm_S(palm, return_spatial=False)             # [B,256]
            fv_S = cnn_vein_S(vein, return_spatial=False)             # [B,256]
            z_S = fusion_S(fp_S, fv_S)
            logit_S = classifier_S(z_S, y)

            loss_cls = ce(logit_S, y)
            loss_cwd_palm = cwd_loss(feat_palm_S, feat_palm_T.detach())
            loss_cwd_vein = cwd_loss(feat_vein_S, feat_vein_T.detach())
            loss_cwd = 0.5 * (loss_cwd_palm + loss_cwd_vein)

            loss = args.lambda_cls * loss_cls + args.lambda_cwd * loss_cwd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnn_palm_S.parameters()) +
                list(cnn_vein_S.parameters()) +
                list(fusion_S.parameters()) +
                list(classifier_S.parameters()),
                1.0
            )
            optimizer.step()

            bs = palm.size(0)
            epoch_loss += loss.item() * bs
            epoch_cls += loss_cls.item() * bs
            epoch_cwd += loss_cwd.item() * bs
            correct += (logit_S.argmax(1) == y).sum().item()
            seen += bs

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
                cwd=f"{loss_cwd.item():.4f}"
            )

        avg_loss = epoch_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={avg_loss:.4f} acc={train_acc*100:.2f}%")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/loss_cls", epoch_cls / max(seen, 1), epoch)
        writer.add_scalar("train/loss_cwd", epoch_cwd / max(seen, 1), epoch)
        writer.add_scalar("train/acc", train_acc, epoch)

        # ---- eval every N epochs ----
        if epoch % args.eval_every == 0:
            eer, tar_list, warn, v_loss, v_acc = evaluate_eer_tar(
                cnn_palm_S, cnn_vein_S, fusion_S, val_loader, device, args.far_list,
                classifier=classifier_S, ce_fn=ce, desc=f"Val@{epoch}"
            )
            writer.add_scalar("val/loss", v_loss, epoch)
            writer.add_scalar("val/acc", v_acc, epoch)

            if warn:
                print(warn)
                print(f"[VAL] Epoch {epoch}: loss={v_loss:.4f} acc={v_acc*100:.2f}% EER=NaN")
            else:
                tar_str = " ".join([f"TAR@FAR={far:.0e}:{tar*100:.2f}%" for far, tar in tar_list])
                writer.add_scalar("val/EER", eer, epoch)
                for far, tar in tar_list:
                    writer.add_scalar(f"val/TAR@FAR_{far:.0e}", tar, epoch)
                print(f"[VAL] Epoch {epoch}: loss={v_loss:.4f} acc={v_acc*100:.2f}% EER={eer*100:.2f}% | {tar_str}")

                if eer < best_eer:
                    best_eer = eer
                    torch.save({
                        "cnn_palm": cnn_palm_S.state_dict(),
                        "cnn_vein": cnn_vein_S.state_dict(),
                        "fusion": fusion_S.state_dict(),
                        "classifier": classifier_S.state_dict(),
                        "epoch": epoch,
                        "best_eer": best_eer
                    }, best_path)
                    print(f"[SAVE] best_eer={best_eer*100:.2f}% -> {best_path}")

    last_path = os.path.join(args.save_dir, "student_last_cwd.pth")
    torch.save({
        "cnn_palm": cnn_palm_S.state_dict(),
        "cnn_vein": cnn_vein_S.state_dict(),
        "fusion": fusion_S.state_dict(),
        "classifier": classifier_S.state_dict(),
        "epoch": args.epochs,
        "best_eer": best_eer
    }, last_path)
    writer.close()
    print(f"[DONE] last={last_path}, best={best_path}")


if __name__ == "__main__":
    main()
