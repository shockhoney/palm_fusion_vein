import os
import sys
import argparse
import importlib

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

def get_config():
    class Config:
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        input_size = 224
        batch_size = 16
        num_workers = 4
        nir_list = os.path.join(project_root, "data_txt/PolyU_palmvein_list.txt")
        red_list = os.path.join(project_root, "data_txt/PolyU_palmprint_list.txt")
        phase2_pair_txt = os.path.join(project_root, "data_txt/polyu_phase2_test.txt")
        backbone = 'mobilefacenet'
        palm_ckpt_path = os.path.join(project_root, "outputs/polyu_models/cnn_palm_phase1_best_demo.pth")
        vein_ckpt_path = os.path.join(project_root, "outputs/polyu_models/cnn_vein_phase1_best_demo.pth")
    return Config()

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
def extract_fusion_features(cnn_palm: nn.Module, cnn_vein: nn.Module, fusion_model: nn.Module, loader: DataLoader, device: str, method: str):
    cnn_palm.eval()
    cnn_vein.eval()
    fusion_model.eval()

    feats, labels = [], []

    for palm_img, vein_img, labs in tqdm(loader, desc=f"Extract {method} Fusion features"):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)

        is_spatial = (method == 'BPFNet')

        palm_feats = cnn_palm(palm_img, return_spatial=is_spatial)                 
        vein_feats = cnn_vein(vein_img, return_spatial=is_spatial)                 

        fused = fusion_model(palm_feats, vein_feats)
        fused = F.normalize(fused, dim=1)

        feats.append(fused.cpu().numpy())
        labels.append(labs.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Test Fusion Modules')
    parser.add_argument('--method', type=str, required=True, choices=['BPFNet', 'HF', 'MCF', 'MIBFL', 'SS', 'DWS'],
                        help='Which fusion method to test')
    args = parser.parse_args()
    
    method = args.method
    cfg = get_config()
    device = cfg.device

    print(f"[{method}] Building networks and loading checkpoint...")
    
    # Import build_backbone 
    from train_teacher_BPFNet import build_backbone, get_transforms
    cnn_palm, feat_dim, local_dim = build_backbone(cfg.backbone) 
    cnn_vein, _, _ = build_backbone(cfg.backbone)

    stage2_ckpt_path = os.path.join(project_root, f"outputs/polyu_models/stage2_{method}_best.pth")

    if not os.path.exists(cfg.palm_ckpt_path) or not os.path.exists(cfg.vein_ckpt_path):
        raise FileNotFoundError(f"Ensure Phase 1 weights exist: {cfg.palm_ckpt_path}")

    cnn_palm.load_state_dict(torch.load(cfg.palm_ckpt_path, map_location=device)['model'])
    cnn_vein.load_state_dict(torch.load(cfg.vein_ckpt_path, map_location=device)['model'])

    fusion_model = None
    if method != 'HF':
        if method == 'BPFNet':
            from train_teacher_BPFNet import Stage2FusionBPFNet
            fusion_model = Stage2FusionBPFNet(in_dim_global=256, out_dim_final=512).to(device)
        elif method == 'MCF':
            from train_teacher_MCF import Stage2FusionMCF
            fusion_model = Stage2FusionMCF(in_dim=feat_dim, final_l2norm=True).to(device)
        elif method == 'MIBFL':
            from train_teacher_MIBFL import Stage2FusionMIBFL
            fusion_model = Stage2FusionMIBFL(in_dim=feat_dim, hash_len=512).to(device)
        elif method == 'SS':
            from train_teacher_SS import Stage2FusionSS
            fusion_model = Stage2FusionSS().to(device)
        elif method == 'DWS':
            from train_teacher_DWS import Stage2FusionDWS
            fusion_model = Stage2FusionDWS(in_dim=feat_dim).to(device)

        if not os.path.exists(stage2_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {stage2_ckpt_path}")
        
        ckpt = torch.load(stage2_ckpt_path, map_location=device)
        if "fusion" in ckpt:
            fusion_model.load_state_dict(ckpt["fusion"])
            print(f"[{method}] Loaded Fusion module weights successfully.")
        else:
            raise KeyError("checkpoint must include 'fusion'")
    else:
        # HF is score-level fusion, no extract_fusion_features needed
        if not os.path.exists(stage2_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {stage2_ckpt_path}")
        ckpt = torch.load(stage2_ckpt_path, map_location=device)
        hf_weights = torch.softmax(ckpt["fusion"]["weights"], dim=0).detach().cpu().numpy()
        print(f"[{method}] Loaded HF Score weights successfully: Palm={hf_weights[0]:.4f}, Vein={hf_weights[1]:.4f}")

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

    print(f"\nEvaluating Multi-modal Fusion ({method})...")
    if os.path.exists(cfg.phase2_pair_txt):
        pair_dataset = PairTxtDataset(cfg.phase2_pair_txt, transform_palm=tf_test, transform_vein=tf_test)
        pair_loader = DataLoader(pair_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        if method == 'HF':
            p_feats, p_labels = [], []
            v_feats, v_labels = [], []
            for palm_img, vein_img, labs in tqdm(pair_loader, desc=f"Extract {method} separate features"):
                palm_img, vein_img = palm_img.to(device), vein_img.to(device)
                pf = F.normalize(cnn_palm(palm_img, return_spatial=False), dim=1)
                vf = F.normalize(cnn_vein(vein_img, return_spatial=False), dim=1)
                p_feats.append(pf.detach().cpu().numpy())
                v_feats.append(vf.detach().cpu().numpy())
                p_labels.append(labs.numpy())
            p_feats = np.concatenate(p_feats, axis=0)
            v_feats = np.concatenate(v_feats, axis=0)
            labels_arr = np.concatenate(p_labels, axis=0)
            
            p_scores, fused_pair_labels = build_pair_scores(p_feats, labels_arr)
            v_scores, _ = build_pair_scores(v_feats, labels_arr)
            
            fused_scores = hf_weights[0] * p_scores + hf_weights[1] * v_scores
            eval_with_metrics(fused_scores, fused_pair_labels, name=f"Phase2 - {method} Fusion (NIR+Red)")
        else:
            fused_feats, fused_labels = extract_fusion_features(cnn_palm, cnn_vein, fusion_model, pair_loader, device, method)
            fused_scores, fused_pair_labels = build_pair_scores(fused_feats, fused_labels)
            
            eval_with_metrics(fused_scores, fused_pair_labels, name=f"Phase2 - {method} Fusion (NIR+Red)")
    else:
        print(f"Warning: '{cfg.phase2_pair_txt}' not found")

if __name__ == "__main__":
    main()
