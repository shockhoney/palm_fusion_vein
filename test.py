import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.datasets_txt import TxtImageDataset, PairTxtDataset
from models.stage2 import Stage2Fusion

from utils.metrics import compute_eer,roc_auc,tar_at_far,far_frr_acc_at_threshold
from train import build_backbone,get_transforms

class Config:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "mobilefacenet"  
    input_size = 224
    batch_size = 32
    num_workers = 4
    nir_list = "polyu__NIR_list.txt"
    red_list = "polyu__Red_list.txt"
    phase2_pair_txt = "phase2_test_pairs.txt"
    backbone = 'mobilefacenet'  # 'convnext' or 'mobilefacenet'
    stage2_ckpt = os.path.join("outputs", "models", "stage2_best.pth")


@torch.no_grad()
def extract_global_features(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    feats, labels = [], []

    for imgs, labs in tqdm(loader, desc="Extract features"):
        imgs = imgs.to(device)
        emb = model(imgs)                 
        emb = F.normalize(emb, dim=1)    
        feats.append(emb.cpu().numpy())
        labels.append(labs.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

def build_pair_scores(features, labels):
    """
    pair_labels = 1 表示同一人,0 表示不同人
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
def extract_fusion_features(cnn_palm: nn.Module,cnn_vein: nn.Module,fusion_model: nn.Module,loader: DataLoader,device: str):

    cnn_palm.eval()
    cnn_vein.eval()
    fusion_model.eval()

    feats, labels = [], []

    for palm_img, vein_img, labs in tqdm(loader, desc="Extract fusion features"):
        palm_img = palm_img.to(device)
        vein_img = vein_img.to(device)

        palm_global = cnn_palm(palm_img)                 
        vein_global = cnn_vein(vein_img)                 

        palm_local = cnn_palm(palm_img, return_spatial=True)
        vein_local = cnn_vein(vein_img, return_spatial=True)

        fused = fusion_model(palm_global, vein_global, palm_local, vein_local)
        fused = F.normalize(fused, dim=1)

        feats.append(fused.cpu().numpy())
        labels.append(labs.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    cfg = Config()
    device = cfg.device

    cnn_palm, feat_dim, local_dim = build_backbone(cfg.backbone)
    cnn_vein, _, _ = build_backbone(cfg.backbone)
    fusion_model = Stage2Fusion(
        in_dim_global_palm=feat_dim,
        in_dim_global_vein=feat_dim,
        in_dim_local_palm=local_dim,
        in_dim_local_vein=local_dim,
        out_dim_local=min(256, local_dim),
        final_l2norm=True,
        out_dim_final=512).to(device)

    # 只需加载第二阶段checkpoint，里面含有第一阶段的模型权重和融合模型权重
    ckpt = torch.load(cfg.stage2_ckpt, map_location=device)
    if all(k in ckpt for k in ("cnn_palm", "cnn_vein", "fusion")):
        cnn_palm.load_state_dict(ckpt["cnn_palm"])
        cnn_vein.load_state_dict(ckpt["cnn_vein"])
        fusion_model.load_state_dict(ckpt["fusion"])
    else:
        raise KeyError("checkpoint must include 'cnn_palm', 'cnn_vein', 'fusion'")

    tf_test = get_transforms(cfg.input_size,strong=False)
    nir_test = TxtImageDataset(cfg.nir_list, split="test", transform=tf_test)
    red_test = TxtImageDataset(cfg.red_list, split="test", transform=tf_test)

    nir_loader = DataLoader( nir_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    red_loader = DataLoader(red_test,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

    nir_feats, nir_labels = extract_global_features(cnn_vein, nir_loader, device)
    red_feats, red_labels = extract_global_features(cnn_palm, red_loader, device)

    nir_scores, nir_pair_labels = build_pair_scores(nir_feats, nir_labels)
    red_scores, red_pair_labels = build_pair_scores(red_feats, red_labels)

    eval_with_metrics(nir_scores, nir_pair_labels, name="Phase1 - NIR (vein) only")
    eval_with_metrics(red_scores, red_pair_labels, name="Phase1 - Red (palm) only")

    if os.path.exists(cfg.phase2_pair_txt):
        pair_dataset = PairTxtDataset(cfg.phase2_pair_txt,transform_palm=tf_test,transform_vein=tf_test)
        pair_loader = DataLoader( pair_dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

        fused_feats, fused_labels = extract_fusion_features(cnn_palm, cnn_vein, fusion_model, pair_loader, device)
        fused_scores, fused_pair_labels = build_pair_scores(fused_feats, fused_labels)
        eval_with_metrics(fused_scores, fused_pair_labels, name="Phase2 - Fusion (NIR+Red)")
    else:
        print(f"Warning: '{cfg.phase2_pair_txt}' not found")


if __name__ == "__main__":
    main()
