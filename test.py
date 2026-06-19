import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.stage2 import Stage2Fusion
from models.student_fusion import Stage2FusionStudent_BottleneckGate
from train_teacher import build_backbone, config as train_config
from utils.datasets_txt import PairTxtDataset, TxtImageDataset
from utils.metrics import compute_eer, far_frr_acc_at_threshold, roc_auc, tar_at_far


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def extract_global_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, labs in tqdm(loader, desc="Extract features", dynamic_ncols=True):
        emb = F.normalize(model(imgs.to(device)), dim=1)
        feats.append(emb.cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def build_pair_scores(features, labels):
    sim = np.asarray(features) @ np.asarray(features).T
    labels = np.asarray(labels)
    i, j = np.triu_indices(labels.shape[0], k=1)
    return sim[i, j], (labels[i] == labels[j]).astype(int), i, j


def build_fusion_model(fusion_type, backbone, feat_dim, ckpt):
    if fusion_type == "auto":
        fusion_type = ckpt.get("student_fusion")
        if fusion_type is None:
            fusion_type = "bottleneck_gate" if backbone == "mobilefacenet" else "stage2"

    if fusion_type == "bottleneck_gate":
        return Stage2FusionStudent_BottleneckGate(in_dim_global=feat_dim, out_dim_final=512, final_l2norm=True)
    return Stage2Fusion(in_dim_global=feat_dim, out_dim_final=512, final_l2norm=True)


def eval_with_metrics(scores, pair_labels, name, out_csv=None):
    eer, thr = compute_eer(scores, pair_labels, is_similarity=True, return_threshold=True)
    _, _, _, auc_val = roc_auc(scores, pair_labels, is_similarity=True)
    thr_stats = far_frr_acc_at_threshold(scores, pair_labels, thr, is_similarity=True)
    fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    tar_info = {far: tar_at_far(scores, pair_labels, far, is_similarity=True) for far in fars}

    print(f"\n===== {name} =====")
    print(f"AUC : {auc_val:.4f}")
    print(f"EER : {eer * 100:.3f}% (threshold = {thr:.4f})")
    print(f"ACC@EER_thr = {thr_stats['ACC']:.4f}, FAR={thr_stats['FAR']:.4f}, FRR={thr_stats['FRR']:.4f}")
    for far, info in tar_info.items():
        print(f"  FAR={far:.1e}: TAR={info['TAR']:.4f}, thr={info['threshold']:.4f}")

    if out_csv:
        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["name", "auc", "eer", "acc_at_eer", "tar_1e-5", "tar_1e-4", "tar_1e-3"])
            writer.writerow([name, auc_val, eer, thr_stats["ACC"], tar_info[1e-5]["TAR"], tar_info[1e-4]["TAR"], tar_info[1e-3]["TAR"]])
    return thr


@torch.no_grad()
def extract_fusion_features(cnn_palm, cnn_vein, fusion_model, loader, device):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion_model.eval()
    feats, labels = [], []
    for palm_img, vein_img, labs in tqdm(loader, desc="Extract fusion features", dynamic_ncols=True):
        palm_feat = cnn_palm(palm_img.to(device), return_spatial=False)
        vein_feat = cnn_vein(vein_img.to(device), return_spatial=False)
        fused = F.normalize(fusion_model(palm_feat, vein_feat), dim=1)
        feats.append(fused.cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def export_failures(path, dataset, scores, labels, i_idx, j_idx, threshold, top_k):
    rows = []
    preds = scores >= threshold
    for n, (score, label, pred) in enumerate(zip(scores, labels, preds)):
        if int(label) == int(pred):
            continue
        a, b = int(i_idx[n]), int(j_idx[n])
        p1, v1, y1 = dataset.samples[a]
        p2, v2, y2 = dataset.samples[b]
        err = "false_accept" if pred else "false_reject"
        rows.append((err, abs(float(score - threshold)), float(score), int(label), y1, y2, p1, v1, p2, v2))
    rows.sort(key=lambda x: x[1], reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error", "margin", "score", "pair_label", "id1", "id2", "palm1", "vein1", "palm2", "vein2"])
        writer.writerows(rows[:top_k])


def main():
    parser = argparse.ArgumentParser("Evaluate palmprint/palm-vein verification")
    parser.add_argument("--ckpt", default="outputs/polyu_models_42/student_last_distill.pth")
    parser.add_argument("--backbone", default="mobilefacenet", choices=["mobilefacenet", "resnet18"])
    parser.add_argument("--fusion", default="auto", choices=["auto", "stage2", "bottleneck_gate"])
    parser.add_argument("--palm_list", default="data_txt/PolyU_palmprint_list.txt")
    parser.add_argument("--vein_list", default="data_txt/PolyU_palmvein_list.txt")
    parser.add_argument("--pair_txt", default="data_txt/polyu_phase2_test.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--failure_csv", default=None)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_config.device = str(device)
    ckpt = safe_torch_load(args.ckpt, device)
    cnn_palm, feat_dim, _ = build_backbone(args.backbone)
    cnn_vein, _, _ = build_backbone(args.backbone)
    cnn_palm.to(device)
    cnn_vein.to(device)
    fusion_model = build_fusion_model(args.fusion, args.backbone, feat_dim, ckpt).to(device)

    cnn_palm.load_state_dict(ckpt["cnn_palm"])
    cnn_vein.load_state_dict(ckpt["cnn_vein"])
    fusion_model.load_state_dict(ckpt["fusion"])

    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    palm_set = TxtImageDataset(args.palm_list, split="test", transform=tf_test)
    vein_set = TxtImageDataset(args.vein_list, split="test", transform=tf_test)
    palm_loader = DataLoader(palm_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    vein_loader = DataLoader(vein_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vein_feats, vein_labels = extract_global_features(cnn_vein, vein_loader, device)
    palm_feats, palm_labels = extract_global_features(cnn_palm, palm_loader, device)
    vein_scores, vein_pair_labels, _, _ = build_pair_scores(vein_feats, vein_labels)
    palm_scores, palm_pair_labels, _, _ = build_pair_scores(palm_feats, palm_labels)
    eval_with_metrics(vein_scores, vein_pair_labels, "Palm-vein only", args.out_csv)
    eval_with_metrics(palm_scores, palm_pair_labels, "Palmprint only", args.out_csv)

    pair_set = PairTxtDataset(args.pair_txt, transform_palm=tf_test, transform_vein=tf_test)
    pair_loader = DataLoader(pair_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    fused_feats, fused_labels = extract_fusion_features(cnn_palm, cnn_vein, fusion_model, pair_loader, device)
    fused_scores, fused_pair_labels, i_idx, j_idx = build_pair_scores(fused_feats, fused_labels)
    threshold = eval_with_metrics(fused_scores, fused_pair_labels, "Fusion", args.out_csv)
    if args.failure_csv:
        export_failures(args.failure_csv, pair_set, fused_scores, fused_pair_labels, i_idx, j_idx, threshold, args.top_k)


if __name__ == "__main__":
    main()
