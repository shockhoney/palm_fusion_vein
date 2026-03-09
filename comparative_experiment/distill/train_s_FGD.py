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
# FGD Loss (CVPR 2022) adapted for classification
# Official repo: yzd-v/FGD
# -------------------------
class FeatureLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 alpha_fgd=0.001,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_mask_s.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.conv_mask_t.weight, mode='fan_in')
        nn.init.constant_(self.channel_add_conv_s[-1].weight, 0)
        nn.init.constant_(self.channel_add_conv_t[-1].weight, 0)

    def forward(self, preds_S, preds_T):
        """
        preds_S: BxCxHxW
        preds_T: BxCxHxW
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'Spatial mismatch'

        if self.align is not None:
            preds_S = self.align(preds_S)

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        # Without bounding boxes, we treat the whole image as foreground for attention mimicking.
        Mask_fg = torch.ones_like(S_attention_t)

        fg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        loss = self.alpha_fgd * fg_loss + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        return loss

    def get_attention(self, preds, temp):
        N, C, H, W = preds.shape
        value = torch.abs(preds)
        # Spatial Attention: Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)
        # Channel Attention: Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)
        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        C_t = C_t.unsqueeze(dim=-1).unsqueeze(dim=-1)
        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        return fg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)
        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x.view(batch, channel, height * width).unsqueeze(1)
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = F.softmax(context_mask, dim=2).unsqueeze(-1)
        context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)
        
        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = preds_S + channel_add_s
        
        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = preds_T + channel_add_t
        
        rela_loss = loss_mse(out_s, out_t) / len(out_s)
        return rela_loss

def safe_torch_load(path, device):
    # Avoid torch.load warning in newer PyTorch if possible
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser("Distill student (TinyMobileFaceNet) from teacher (MobileFaceNet)")
    parser.add_argument("--train_list", type=str, default=os.path.join(project_root, "data_txt/polyu_phase2_train.txt"))
    parser.add_argument("--val_list", type=str, default=os.path.join(project_root, "data_txt/polyu_phase2_val.txt"))
    parser.add_argument("--teacher_ckpt", type=str, default=os.path.join(project_root, "outputs/polyu_models/stage2_best.pth"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(project_root, "outputs_distill/FGD_models"))

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    # distill weights
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--alpha_fgd", type=float, default=0.001)
    parser.add_argument("--gamma_fgd", type=float, default=0.001)
    parser.add_argument("--lambda_fgd", type=float, default=0.000005)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs_distill"))

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

    # Instantiate FGD Loss modules for each branch (palm and vein)
    fgd_loss_palm = FeatureLoss(student_channels=feat_dim_S, teacher_channels=feat_dim_T, 
                                temp=args.temp, alpha_fgd=args.alpha_fgd, 
                                gamma_fgd=args.gamma_fgd, lambda_fgd=args.lambda_fgd).to(device)
    fgd_loss_vein = FeatureLoss(student_channels=feat_dim_S, teacher_channels=feat_dim_T, 
                                temp=args.temp, alpha_fgd=args.alpha_fgd, 
                                gamma_fgd=args.gamma_fgd, lambda_fgd=args.lambda_fgd).to(device)

    optimizer_params = (
        list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) + list(classifier_S.parameters()) +
        list(fgd_loss_palm.parameters()) + list(fgd_loss_vein.parameters())
    )
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_fgd.pth")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        fgd_loss_palm.train()
        fgd_loss_vein.train()

        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_fgd = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for palm, vein, y in pbar:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                feat_palm_T = cnn_palm_T(palm, return_spatial=True)
                feat_vein_T = cnn_vein_T(vein, return_spatial=True)

            # ---- student forward ----
            feat_palm_S = cnn_palm_S(palm, return_spatial=True)
            feat_vein_S = cnn_vein_S(vein, return_spatial=True)

            fp_S = cnn_palm_S.bn(cnn_palm_S.global_pool(feat_palm_S).flatten(1))
            fv_S = cnn_vein_S.bn(cnn_vein_S.global_pool(feat_vein_S).flatten(1))

            z_S = fusion_S(fp_S, fv_S)

            logit_S = classifier_S(z_S, y)
            loss_cls = ce(logit_S, y)

            # FGD loss on spatial features
            loss_fgd_p = fgd_loss_palm(feat_palm_S, feat_palm_T)
            loss_fgd_v = fgd_loss_vein(feat_vein_S, feat_vein_T)
            loss_fgd = 0.5 * (loss_fgd_p + loss_fgd_v)

            loss = args.lambda_cls * loss_cls + loss_fgd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
            optimizer.step()

            bs = palm.size(0)
            epoch_loss += loss.item() * bs
            epoch_cls += loss_cls.item() * bs
            epoch_fgd += loss_fgd.item() * bs
            correct += (logit_S.argmax(1) == y).sum().item()
            seen += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", fgd=f"{loss_fgd.item():.4f}")

        avg_loss = epoch_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={avg_loss:.4f} acc={train_acc*100:.2f}%")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/loss_cls", epoch_cls / max(seen, 1), epoch)
        writer.add_scalar("train/loss_fgd", epoch_fgd / max(seen, 1), epoch)
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

    last_path = os.path.join(args.save_dir, "student_last_fgd.pth")
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
