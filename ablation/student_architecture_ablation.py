import os
import argparse
import random
import subprocess
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.datasets_txt import PairTxtDataset, TxtImageDataset
from utils.metrics import compute_eer, tar_at_far
from utils.head import Arcface_Head

from models.stage1_mobileFacenet import MobileFaceNet
from models.resnet18_encoder import ResNet18Encoder
from models.stage2 import Stage2Fusion
from models.student_fusion import Stage2FusionStudent_BottleneckGate
from test import (
    build_pair_scores,
    eval_with_metrics,
    export_failures,
    extract_fusion_features,
    extract_global_features,
    safe_torch_load,
)

ARCH_VARIANTS = ("mobile_concat", "mobile_eca_concat", "mobile_gate", "mobile_eca_gate")


class MobileFaceNetAblation(MobileFaceNet):
    def __init__(self, *args, use_eca=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not use_eca:
            self.eca = nn.Identity()


class ConcatLinearFusion(nn.Module):
    def __init__(self, in_dim_global=256, out_dim_final=512, final_l2norm=True):
        super().__init__()
        self.final_l2norm = final_l2norm
        self.proj = nn.Linear(2 * in_dim_global, out_dim_final)

    def forward(self, palm, vein):
        fused = self.proj(torch.cat([palm, vein], dim=1))
        return F.normalize(fused, dim=1) if self.final_l2norm else fused


def evaluate_checkpoint(args):
    device = torch.device(args.device)
    checkpoint = safe_torch_load(args.ckpt, device)
    use_eca = args.variant in {"mobile_eca_concat", "mobile_eca_gate"}
    use_gate = args.variant in {"mobile_gate", "mobile_eca_gate"}
    palm_net = MobileFaceNetAblation(input_channel=3, input_size=224, use_eca=use_eca).to(device)
    vein_net = MobileFaceNetAblation(input_channel=3, input_size=224, use_eca=use_eca).to(device)
    fusion = (
        Stage2FusionStudent_BottleneckGate(in_dim_global=256, out_dim_final=512, bottleneck=128, gate_hidden=32, final_l2norm=True)
        if use_gate else ConcatLinearFusion(in_dim_global=256, out_dim_final=512, final_l2norm=True)
    ).to(device)
    palm_net.load_state_dict(checkpoint["cnn_palm"])
    vein_net.load_state_dict(checkpoint["cnn_vein"])
    fusion.load_state_dict(checkpoint["fusion"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    palm_set = TxtImageDataset(args.palm_list, split="test", transform=transform)
    vein_set = TxtImageDataset(args.vein_list, split="test", transform=transform)
    palm_loader = DataLoader(palm_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)
    vein_loader = DataLoader(vein_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)
    palm_feats, palm_labels = extract_global_features(palm_net, palm_loader, device)
    vein_feats, vein_labels = extract_global_features(vein_net, vein_loader, device)
    palm_scores, palm_pair_labels = build_pair_scores(palm_feats, palm_labels)[:2]
    vein_scores, vein_pair_labels = build_pair_scores(vein_feats, vein_labels)[:2]
    eval_with_metrics(palm_scores, palm_pair_labels, "Palmprint only", args.out_csv)
    eval_with_metrics(vein_scores, vein_pair_labels, "Palm-vein only", args.out_csv)

    pair_set = PairTxtDataset(args.pair_txt, transform_palm=transform, transform_vein=transform)
    pair_loader = DataLoader(pair_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers)
    fused_feats, fused_labels = extract_fusion_features(palm_net, vein_net, fusion, pair_loader, device)
    pair_result = build_pair_scores(fused_feats, fused_labels)
    fused_scores, fused_pair_labels = pair_result[:2]
    if len(pair_result) == 4:
        i_idx, j_idx = pair_result[2:]
    else:
        i_idx, j_idx = np.triu_indices(len(fused_labels), k=1)
    threshold = eval_with_metrics(fused_scores, fused_pair_labels, f"Fusion/{args.variant}", args.out_csv)
    if args.failure_csv:
        export_failures(args.failure_csv, pair_set, fused_scores, fused_pair_labels, i_idx, j_idx, threshold, args.top_k)

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
        msg = f"[WARN] Validation pairs invalid for EER/TAR: pos_pairs={pos}, neg_pairs={neg}."
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
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--variant", choices=[*ARCH_VARIANTS, "all"], required=True)
    parser.add_argument("--train_list", type=str, default="data_txt/polyu_phase2_train.txt")
    parser.add_argument("--val_list", type=str, default="data_txt/polyu_phase2_val.txt")
    parser.add_argument("--teacher_ckpt", type=str, default="outputs/polyu_models/stage2_best.pth")
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    # distill weights
    parser.add_argument("--lambda_emb", type=float, default=2.0)
    parser.add_argument("--lambda_rel", type=float, default=2.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--ramp_epochs", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--palm_list", type=str, default=None)
    parser.add_argument("--vein_list", type=str, default=None)
    parser.add_argument("--pair_txt", type=str, default=None)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--test_num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--failure_csv", default=None)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "test":
        if args.variant == "all":
            parser.error("Test one variant and checkpoint at a time")
        for name in ("ckpt", "palm_list", "vein_list", "pair_txt"):
            if not getattr(args, name):
                parser.error(f"--{name} is required in test mode")
        evaluate_checkpoint(args)
        return

    if args.variant == "all":
        if args.run_name:
            parser.error("--run_name cannot be used with --variant all")
        variant_index = sys.argv.index("--variant")
        for variant in ARCH_VARIANTS:
            child_args = sys.argv[1:].copy()
            child_args[variant_index - 1:variant_index + 1] = ["--variant", variant]
            subprocess.run([sys.executable, __file__, *child_args], cwd=ROOT, check=True)
        return

    run_name = args.run_name or f"student_arch_{args.variant}_seed{args.seed}"
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
    # Student architecture ablation
    # -------------------------
    use_eca = args.variant in {"mobile_eca_concat", "mobile_eca_gate"}
    use_gate = args.variant in {"mobile_gate", "mobile_eca_gate"}
    cnn_palm_S = MobileFaceNetAblation(input_channel=3, input_size=224, use_eca=use_eca).to(device)
    cnn_vein_S = MobileFaceNetAblation(input_channel=3, input_size=224, use_eca=use_eca).to(device)
    feat_dim_S = cnn_palm_S.out_dim

    if use_gate:
        fusion_S = Stage2FusionStudent_BottleneckGate(
            in_dim_global=feat_dim_S, out_dim_final=512, bottleneck=128, gate_hidden=32, final_l2norm=True
        ).to(device)
    else:
        fusion_S = ConcatLinearFusion(in_dim_global=feat_dim_S, out_dim_final=512, final_l2norm=True).to(device)
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
                        "student_fusion": "bottleneck_gate" if use_gate else "concat_linear",
                        "student_arch": args.variant,
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
        "student_fusion": "bottleneck_gate" if use_gate else "concat_linear",
        "student_arch": args.variant,
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
