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

from sem_CKD_student_mobilefacenet import MobileFaceNet, TinyMobileFaceNet
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
# SemCKD Modules (AAAI 2021)
# -------------------------
class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class SelfA(nn.Module):
    """Cross layer Self Attention"""
    def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4): 
        super(SelfA, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(t_len):
            setattr(self, 'key_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        for i in range(s_len):
            setattr(self, 'query_weight'+str(i), MLPEmbed(input_channel, input_channel//factor))
        
        for i in range(s_len):
            for j in range(t_len):
                setattr(self, 'regressor'+str(i)+str(j), AAEmbed(s_n[i], s_t[j]))
               
    def forward(self, feat_s, feat_t):
        
        sim_t = list(range(len(feat_t)))
        sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        # similarity matrix
        for i in range(len(feat_t)):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(len(feat_s)):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # key of target layers    
        proj_key = getattr(self, 'key_weight0')(sim_t[0])
        proj_key = proj_key[:, :, None]
        
        for i in range(1, len(sim_t)):
            temp_proj_key = getattr(self, 'key_weight'+str(i))(sim_t[i])
            proj_key =  torch.cat([proj_key, temp_proj_key[:, :, None]], 2)
        
        # query of source layers   
        proj_query = getattr(self, 'query_weight0')(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, len(sim_s)):
            temp_proj_query = getattr(self, 'query_weight'+str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)
        
        # attention weight
        energy = torch.bmm(proj_query, proj_key) # batch_size X No.stu feature X No.tea feature
        attention = F.softmax(energy, dim = -1)
        
        # feature space alignment
        proj_value_stu = []
        value_tea = []
        for i in range(len(sim_s)):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(len(sim_t)):            
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    input = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(input))
                    value_tea[i].append(feat_t[j])
                elif s_H < t_H or s_H == t_H:
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                    proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(feat_s[i]))
                    value_tea[i].append(target)
                
        return proj_value_stu, value_tea, attention


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""
    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')
        
    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea, device=weight.device)

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz,-1).mean(-1)

        loss = (weight * ind_loss).sum()/(1.0*bsz*num_stu)
        return loss


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
    parser.add_argument("--save_dir", type=str, default=os.path.join(project_root, "outputs_distill/sem-CKD_models"))

    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--far_list", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])

    # distill weights
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_semckd", type=float, default=10.0)

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

    # Note: the architecture channels below are manually matched
    # Teacher (MobileFaceNet): conv3(32)-conv4(64)-conv5(64)-conv6(256)
    # Student (TinyMobileFaceNet): conv3(16)-conv4(32)-conv5(32)-conv6(256)
    t_n = [32, 64, 64, 256]
    s_n = [16, 32, 32, 256]
    
    criterion_kd = SemCKDLoss().to(device)
    selfa_palm = SelfA(len(s_n), len(t_n), args.batch_size, s_n, t_n).to(device)
    selfa_vein = SelfA(len(s_n), len(t_n), args.batch_size, s_n, t_n).to(device)

    optimizer_params = (
        list(cnn_palm_S.parameters()) + list(cnn_vein_S.parameters()) +
        list(fusion_S.parameters()) + list(classifier_S.parameters()) +
        list(selfa_palm.parameters()) + list(selfa_vein.parameters())
    )

    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best_eer = 1e9
    best_path = os.path.join(args.save_dir, "student_best_semckd.pth")

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        cnn_palm_S.train()
        cnn_vein_S.train()
        fusion_S.train()
        classifier_S.train()

        selfa_palm.train()
        selfa_vein.train()

        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_semckd = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for palm, vein, y in pbar:
            palm = palm.to(device, non_blocking=True)
            vein = vein.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- teacher forward (no grad) ----
            with torch.no_grad():
                feat_t_palm = cnn_palm_T(palm, return_all_spatial=True)
                feat_t_vein = cnn_vein_T(vein, return_all_spatial=True)
                # extract global embedding correctly to compute conf if need be (SemCKD generally doesn't)

            # ---- student forward ----
            feat_s_palm = cnn_palm_S(palm, return_all_spatial=True)
            feat_s_vein = cnn_vein_S(vein, return_all_spatial=True)

            fp_S = cnn_palm_S.bn(cnn_palm_S.global_pool(feat_s_palm[-1]).flatten(1))
            fv_S = cnn_vein_S.bn(cnn_vein_S.global_pool(feat_s_vein[-1]).flatten(1))
            
            z_S = fusion_S(fp_S, fv_S)
            logit_S = classifier_S(z_S, y)
            loss_cls = ce(logit_S, y)

            # SemCKD KD calculation
            proj_val_stu_p, val_tea_p, attn_p = selfa_palm(feat_s_palm, feat_t_palm)
            loss_semckd_palm = criterion_kd(proj_val_stu_p, val_tea_p, attn_p)

            proj_val_stu_v, val_tea_v, attn_v = selfa_vein(feat_s_vein, feat_t_vein)
            loss_semckd_vein = criterion_kd(proj_val_stu_v, val_tea_v, attn_v)

            loss_semckd = 0.5 * (loss_semckd_palm + loss_semckd_vein)

            loss = args.lambda_cls * loss_cls + args.lambda_semckd * loss_semckd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
            optimizer.step()

            bs = palm.size(0)
            epoch_loss += loss.item() * bs
            epoch_cls += loss_cls.item() * bs
            epoch_semckd += loss_semckd.item() * bs
            correct += (logit_S.argmax(1) == y).sum().item()
            seen += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", kd_s=f"{loss_semckd.item():.4f}")

        avg_loss = epoch_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)
        print(f"Epoch [{epoch}/{args.epochs}] avg_loss={avg_loss:.4f} acc={train_acc*100:.2f}% | SemCKD l={epoch_semckd/max(seen, 1):.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/loss_cls", epoch_cls / max(seen, 1), epoch)
        writer.add_scalar("train/loss_semckd", epoch_semckd / max(seen, 1), epoch)
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

    last_path = os.path.join(args.save_dir, "student_last_semckd.pth")
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
