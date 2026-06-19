import os
import argparse
import random
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
from models.resnet18_encoder import ResNet18Encoder
from models.stage2 import Stage2Fusion
from models.student_fusion import Stage2FusionStudent_BottleneckGate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


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
# KD losses (concise)
# -------------------------
def cosine_kd_per_sample(z_s, z_t):
    z_s = F.normalize(z_s, dim=1)
    z_t = F.normalize(z_t, dim=1)
    return 1.0 - (z_s * z_t).sum(dim=1)  # (B,)


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


def ramp(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(ramp_epochs))


def safe_torch_load(path, device):
    # Avoid torch.load warning in newer PyTorch if possible
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser("Distill MobileFaceNet student from ResNet18 teacher")
    parser.add_argument("--train_list", type=str, default="data_txt/polyu_phase2_train.txt")
    parser.add_argument("--val_list", type=str, default="data_txt/polyu_phase2_val.txt")
    parser.add_argument("--teacher_ckpt", type=str, default="outputs/polyu_models_42/stage2_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs/polyu_models_42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    # distill weights
    parser.add_argument("--lambda_emb", type=float, default=2.0)
    parser.add_argument("--lambda_rel", type=float, default=2.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--ramp_epochs", type=int, default=20)
    args = parser.parse_args()
    run_name = args.run_name
    args.save_dir = os.path.join(args.save_dir, run_name) if run_name else args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs_distill", run_name or f"seed_{args.seed}"))

    # -------------------------
    # num_classes from train_list
    # -------------------------
    label_set = set()
    with open(args.train_list, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                label_set.add(int(p[2]))
    num_classes = max(label_set) + 1 if label_set else 0
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
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker,
        generator=make_generator(args.seed)
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1)
    )

    # -------------------------
    # Teacher: ResNet18 + Stage2Fusion + classifier (all in ckpt)
    # -------------------------
    cnn_palm_T = ResNet18Encoder(input_channel=3, input_size=224).to(device)
    cnn_vein_T = ResNet18Encoder(input_channel=3, input_size=224).to(device)
    feat_dim_T = cnn_palm_T.out_dim

    fusion_T = Stage2Fusion(in_dim_global=feat_dim_T, out_dim_final=512, final_l2norm=True).to(device)
    classifier_T = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    ckpt = safe_torch_load(args.teacher_ckpt, device)
    if ckpt.get("backbone") and ckpt.get("backbone") != "resnet18":
        print(f"[WARN] teacher checkpoint backbone is {ckpt.get('backbone')}, expected resnet18")
    cnn_palm_T.load_state_dict(ckpt["cnn_palm"], strict=True)
    cnn_vein_T.load_state_dict(ckpt["cnn_vein"], strict=True)
    fusion_T.load_state_dict(ckpt["fusion"], strict=True)
    classifier_T.load_state_dict(ckpt["classifier"], strict=True)

    for m in [cnn_palm_T, cnn_vein_T, fusion_T, classifier_T]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # -------------------------
    # Student: MobileFaceNet + bottleneck-gated fusion + classifier
    # -------------------------
    cnn_palm_S = MobileFaceNet(input_channel=3, input_size=224).to(device)
    cnn_vein_S = MobileFaceNet(input_channel=3, input_size=224).to(device)
    feat_dim_S = cnn_palm_S.out_dim

    fusion_S = Stage2FusionStudent_BottleneckGate(
        in_dim_global=feat_dim_S, out_dim_final=512, bottleneck=128, gate_hidden=32, final_l2norm=True
    ).to(device)
    classifier_S = Arcface_Head(embedding_size=512, num_classes=num_classes, s=30.0, m=0.20).to(device)

    optimizer = torch.optim.AdamW(
        list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) + list(classifier_S.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    ce = nn.CrossEntropyLoss()

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_distill.pth")
    conf_stats_path = os.path.join(args.save_dir, "teacher_confidence_stats.csv")
    val_metrics_path = os.path.join(args.save_dir, "val_metrics.csv")
    with open(conf_stats_path, "w", encoding="utf-8") as f:
        f.write("epoch,count,mean,std,min,p10,p50,p90,max\n")
    with open(val_metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,acc,eer,tar_1e-03,tar_1e-04,tar_1e-05\n")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        w = ramp(epoch, args.ramp_epochs)
        lam_emb = args.lambda_emb * w
        lam_rel = args.lambda_rel * w

        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_emb = 0.0
        epoch_rel = 0.0
        correct = 0
        seen = 0
        conf_values = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for palm, vein, y in pbar:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                fp_T = cnn_palm_T(palm, return_spatial=False)
                fv_T = cnn_vein_T(vein, return_spatial=False)
                z_T = fusion_T(fp_T, fv_T)  # (B,512)
                logit_T = classifier_T(z_T, y)
                conf = F.softmax(logit_T, dim=1).max(dim=1).values.clamp(0.0, 1.0)  # (B,)
                conf_values.append(conf.detach().cpu().numpy())

            # ---- student forward ----
            fp_S = cnn_palm_S(palm, return_spatial=False)
            fv_S = cnn_vein_S(vein, return_spatial=False)
            z_S = fusion_S(fp_S, fv_S)

            logit_S = classifier_S(z_S, y)
            loss_cls = ce(logit_S, y)

            # embedding KD (confidence-weighted)
            emb_per = cosine_kd_per_sample(z_S, z_T)  # (B,)
            loss_emb = (emb_per * conf).sum() / (conf.sum() + 1e-6)

            # relational KD (confidence-weighted)
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
            epoch_loss += loss.item() * bs
            epoch_cls += loss_cls.item() * bs
            epoch_emb += loss_emb.item() * bs
            epoch_rel += loss_rel.item() * bs
            correct += (logit_S.argmax(1) == y).sum().item()
            seen += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", kd_w=f"{w:.2f}", emb=f"{lam_emb:.2f}", rel=f"{lam_rel:.2f}")

        avg_loss = epoch_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        if conf_values:
            conf_epoch = np.concatenate(conf_values)
            with open(conf_stats_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{conf_epoch.size},{conf_epoch.mean():.6f},{conf_epoch.std():.6f},"
                    f"{conf_epoch.min():.6f},{np.percentile(conf_epoch, 10):.6f},"
                    f"{np.percentile(conf_epoch, 50):.6f},{np.percentile(conf_epoch, 90):.6f},"
                    f"{conf_epoch.max():.6f}\n"
                )
            if epoch == args.epochs:
                np.save(os.path.join(args.save_dir, "teacher_confidence_last_epoch.npy"), conf_epoch)
        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={avg_loss:.4f} acc={train_acc*100:.2f}% | kd_w={w:.2f}")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/loss_cls", epoch_cls / max(seen, 1), epoch)
        writer.add_scalar("train/loss_emb", epoch_emb / max(seen, 1), epoch)
        writer.add_scalar("train/loss_rel", epoch_rel / max(seen, 1), epoch)
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
                tar_dict = {far: tar for far, tar in tar_list}
                with open(val_metrics_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},{v_loss:.8f},{v_acc:.8f},{eer:.8f},"
                        f"{tar_dict.get(1e-3, float('nan')):.8f},"
                        f"{tar_dict.get(1e-4, float('nan')):.8f},"
                        f"{tar_dict.get(1e-5, float('nan')):.8f}\n"
                    )
                writer.add_scalar("val/EER", eer, epoch)
                for far, tar in tar_list:
                    writer.add_scalar(f"val/TAR@FAR_{far:.0e}", tar, epoch)
                print(f"[VAL] Epoch {epoch}: loss={v_loss:.4f} acc={v_acc*100:.2f}% EER={eer*100:.2f}% | {tar_str}")

                if eer < best_eer:
                    best_eer = eer
                    torch.save({
                        "teacher_backbone": "resnet18",
                        "teacher_fusion": "stage2",
                        "student_backbone": "mobilefacenet",
                        "student_fusion": "bottleneck_gate",
                        "cnn_palm": cnn_palm_S.state_dict(),
                        "cnn_vein": cnn_vein_S.state_dict(),
                        "fusion": fusion_S.state_dict(),
                        "classifier": classifier_S.state_dict(),
                        "epoch": epoch,
                        "best_eer": best_eer
                    }, best_path)
                    print(f"[SAVE] best_eer={best_eer*100:.2f}% -> {best_path}")

    last_path = os.path.join(args.save_dir, "student_last_distill.pth")
    torch.save({
        "teacher_backbone": "resnet18",
        "teacher_fusion": "stage2",
        "student_backbone": "mobilefacenet",
        "student_fusion": "bottleneck_gate",
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
