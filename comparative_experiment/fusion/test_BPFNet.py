# # 测试 BPFNet
# python test_fusion.py --method BPFNet

# # 测试 HF (Score-level)
# python test_fusion.py --method HF

# # 测试 MCF
# python test_fusion.py --method MCF

# # 测试 MIBFL
# python test_fusion.py --method MIBFL

# # 测试 SS (Simple Sum)
# python test_fusion.py --method SS

# # 测试 DWS (Dynamic Weighted Sum)
# python test_fusion.py --method DWS

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.datasets_txt import TxtImageDataset, PairTxtDataset
from utils.metrics import compute_eer, roc_auc, tar_at_far, far_frr_acc_at_threshold
from train_teacher_BPFNet import build_backbone, get_transforms, Stage2FusionBPFNet

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    input_size = 224
    batch_size = 32
    num_workers = 4
    nir_list = os.path.join(project_root, "data_txt/PolyU_palmvein_list.txt")
    red_list = os.path.join(project_root, "data_txt/PolyU_palmprint_list.txt")
    phase2_pair_txt = os.path.join(project_root, "data_txt/polyu_phase2_test.txt")
    backbone = 'mobilefacenet'

    # Ensure this points to the checkpoint saved by train_teacher_BPFNet.py
    stage2_ckpt = os.path.join(project_root, "outputs/polyu_models/stage2_BPFNet_best.pth")


@torch.no_grad()
def extract_global_features(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    feats, labels = [], []

    for imgs, labs in tqdm(loader, desc="Extract Phase1 features"):
        imgs = imgs.to(device)
        emb = model(imgs, return_spatial=False)                 
        emb = F.normalize(emb, dim=1)    
        feats.append(emb.cpu().numpy())
        labels.append(labs.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

def build_pair_scores(features, labels):
    """
    pair_labels = 1 for same identity, 0 for different
    """
    features = np.asarray(features)
    labels = np.asarray(labels)

    sim = features @ features.T 
    n = labels.shape[0]
    i, j = np.triu_indices(n, k=1)

    scores = sim[i, j]
    pair_labels = (labels[i] == labels[j]).astype(int)
    return scores, pair_labels

def eval_with_metrics(scores, pair_labels, name):
    eer, thr = compute_eer(scores, pair_labels, is_similarity=True, return_threshold=True)
    fpr, tpr, thresholds, auc_val = roc_auc(scores, pair_labels, is_similarity=True)
    thr_stats = far_frr_acc_at_threshold(scores, pair_labels, thr, is_similarity=True)

    fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    tar_info = {far: tar_at_far(scores, pair_labels, far, is_similarity=True) for far in fars}

    print(f"\n===== {name} =====")
    print(f"AUC : {auc_val:.4f}")
    print(f"EER : {eer * 100:.3f}% (threshold = {thr:.4f})")
    print(
        f"ACC@EER_thr = {thr_stats['ACC']:.4f}, "
        f"FAR={thr_stats['FAR']:.4f}, FRR={thr_stats['FRR']:.4f}")
    print("TAR @ FAR:")
    for far, info in tar_info.items():
        print(f"  FAR={far:.1e}: TAR={info['TAR']:.4f}, thr={info['threshold']:.4f}")


@torch.no_grad()
def extract_fusion_features(cnn_palm: nn.Module, cnn_vein: nn.Module, fusion_model: nn.Module, loader: DataLoader, device: str):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion_model.eval()

    feats, labels = [], []

    for palm_img, vein_img, labs in tqdm(loader, desc="Extract BPFNet Fusion features"):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)

        # For BPFNet, we MUST return spatial features
        palm_spatial = cnn_palm(palm_img, return_spatial=True)                 
        vein_spatial = cnn_vein(vein_img, return_spatial=True)                 

        fused = fusion_model(palm_spatial, vein_spatial)
        fused = F.normalize(fused, dim=1)

        feats.append(fused.cpu().numpy())
        labels.append(labs.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    cfg = Config()
    device = cfg.device

    print("Building networks and loading checkpoint...")
    cnn_palm, feat_dim, local_dim = build_backbone(cfg.backbone) # Returns (model, 512, 256)
    cnn_vein, _, _ = build_backbone(cfg.backbone)
    fusion_model = Stage2FusionBPFNet(in_dim_global=256, out_dim_final=512).to(device)

    # 1. Load original Phase 1 trained weights for single modality backbones
    palm_ckpt_path = os.path.join(project_root, "outputs/polyu_models/cnn_palm_phase1_best_demo.pth")
    vein_ckpt_path = os.path.join(project_root, "outputs/polyu_models/cnn_vein_phase1_best_demo.pth")
    
    if os.path.exists(palm_ckpt_path):
        cnn_palm.load_state_dict(torch.load(palm_ckpt_path, map_location=device)['model'])
        print(f"Loaded original Palm Phase 1 weights from: {palm_ckpt_path}")
    else:
        print(f"Warning: Palm Phase 1 checkpoint not found at: {palm_ckpt_path}")

    if os.path.exists(vein_ckpt_path):
        cnn_vein.load_state_dict(torch.load(vein_ckpt_path, map_location=device)['model'])
        print(f"Loaded original Vein Phase 1 weights from: {vein_ckpt_path}")
    else:
        print(f"Warning: Vein Phase 1 checkpoint not found at: {vein_ckpt_path}")

    # 2. Load the Fusion block weights from the joint Phase 2 training
    if not os.path.exists(cfg.stage2_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at: {cfg.stage2_ckpt}")
        
    ckpt = torch.load(cfg.stage2_ckpt, map_location=device)
    if "fusion" in ckpt:
        fusion_model.load_state_dict(ckpt["fusion"])
        print("Loaded Fusion module weights successfully.")
    else:
        raise KeyError("checkpoint must include 'fusion'")

    print("Initializing DataLoaders...")
    tf_test = get_transforms(cfg.input_size, strong=False)
    nir_test = TxtImageDataset(cfg.nir_list, split="test", transform=tf_test)
    red_test = TxtImageDataset(cfg.red_list, split="test", transform=tf_test)

    nir_loader = DataLoader(nir_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    red_loader = DataLoader(red_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print("Evaluating Single Modality...")
    nir_feats, nir_labels = extract_global_features(cnn_vein, nir_loader, device)
    red_feats, red_labels = extract_global_features(cnn_palm, red_loader, device)

    nir_scores, nir_pair_labels = build_pair_scores(nir_feats, nir_labels)
    red_scores, red_pair_labels = build_pair_scores(red_feats, red_labels)

    eval_with_metrics(nir_scores, nir_pair_labels, name="Phase1 - NIR (vein) only")
    eval_with_metrics(red_scores, red_pair_labels, name="Phase1 - Red (palm) only")

    print("\nEvaluating Multi-modal Fusion...")
    if os.path.exists(cfg.phase2_pair_txt):
        pair_dataset = PairTxtDataset(cfg.phase2_pair_txt, transform_palm=tf_test, transform_vein=tf_test)
        pair_loader = DataLoader(pair_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        fused_feats, fused_labels = extract_fusion_features(cnn_palm, cnn_vein, fusion_model, pair_loader, device)
        fused_scores, fused_pair_labels = build_pair_scores(fused_feats, fused_labels)
        
        eval_with_metrics(fused_scores, fused_pair_labels, name="Phase2 - BPFNet Fusion (NIR+Red)")
    else:
        print(f"Warning: '{cfg.phase2_pair_txt}' not found")

if __name__ == "__main__":
    main()
