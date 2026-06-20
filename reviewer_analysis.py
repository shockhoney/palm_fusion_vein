import argparse
import csv
import heapq
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from test import build_backbone, build_fusion_model, build_pair_scores, safe_torch_load, train_config
from utils.datasets_txt import PairTxtDataset
from utils.metrics import compute_eer


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def metrics(scores, labels):
    eer, threshold = compute_eer(scores, labels, return_threshold=True)
    return {"eer": eer, "threshold": threshold}


@torch.no_grad()
def extract_features(models, loader, device):
    palm_net, vein_net, fusion = models
    for model in models:
        model.eval()

    palm_features, vein_features, fusion_features, labels = [], [], [], []
    for palm, vein, batch_labels in tqdm(loader, desc="Extract features", dynamic_ncols=True):
        palm = palm.to(device)
        vein = vein.to(device)
        palm_feature = F.normalize(palm_net(palm, return_spatial=False), dim=1)
        vein_feature = F.normalize(vein_net(vein, return_spatial=False), dim=1)
        fusion_feature = F.normalize(fusion(palm_feature, vein_feature), dim=1)
        palm_features.append(palm_feature.cpu().numpy())
        vein_features.append(vein_feature.cpu().numpy())
        fusion_features.append(fusion_feature.cpu().numpy())
        labels.append(batch_labels.numpy())

    return tuple(np.concatenate(values) for values in (
        palm_features, vein_features, fusion_features, labels
    ))


def confidence_analysis(paths, output_dir):
    rows = []
    plt.figure(figsize=(5, 3.5))
    for path in paths:
        values = np.load(path).astype(float).ravel()
        values = values[np.isfinite(values)]
        percentiles = np.percentile(values, [10, 25, 50, 75, 90])
        rows.append({
            "source": str(path), "count": values.size, "mean": values.mean(),
            "std": values.std(), "min": values.min(), "p10": percentiles[0],
            "p25": percentiles[1], "p50": percentiles[2], "p75": percentiles[3],
            "p90": percentiles[4], "max": values.max(),
        })
        plt.hist(values, bins=40, density=True, alpha=0.45, label=Path(path).parent.name)

    write_csv(output_dir / "confidence_stats.csv", rows)
    plt.xlabel("Teacher confidence")
    plt.ylabel("Density")
    if len(paths) > 1:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=300)
    plt.close()


def failure_analysis(dataset, scores_by_model, labels, indices, metric_rows, output_dir, top_k):
    thresholds = {row["model"]: row["threshold"] for row in metric_rows}
    predictions = {name: scores >= thresholds[name] for name, scores in scores_by_model.items()}
    correct = {name: predictions[name] == labels for name in predictions}
    counts, top_rows = Counter(), defaultdict(list)

    for n, (i, j) in enumerate(zip(*indices)):
        fusion_ok = correct["fusion"][n]
        palm_ok = correct["palmprint"][n]
        vein_ok = correct["palm_vein"][n]
        if fusion_ok and palm_ok and vein_ok:
            counts["all_correct"] += 1
            continue
        if fusion_ok:
            category = "fusion_corrects_both" if not palm_ok and not vein_ok else "fusion_corrects_one"
        elif palm_ok and vein_ok:
            category = "fusion_hurts_both"
        elif palm_ok or vein_ok:
            category = "fusion_wrong_one_unimodal_correct"
        else:
            error = "false_accept" if labels[n] == 0 else "false_reject"
            category = f"all_models_{error}"

        palm1, vein1, id1 = dataset.samples[int(i)]
        palm2, vein2, id2 = dataset.samples[int(j)]
        row = {
            "category": category,
            "severity": abs(scores_by_model["fusion"][n] - thresholds["fusion"]),
            "pair_label": int(labels[n]), "id1": id1, "id2": id2,
            "palmprint_score": scores_by_model["palmprint"][n],
            "palm_vein_score": scores_by_model["palm_vein"][n],
            "fusion_score": scores_by_model["fusion"][n],
            "palmprint_threshold": thresholds["palmprint"],
            "palm_vein_threshold": thresholds["palm_vein"],
            "fusion_threshold": thresholds["fusion"],
            "palmprint_correct": int(palm_ok), "palm_vein_correct": int(vein_ok),
            "fusion_correct": int(fusion_ok),
            "palm1": palm1, "vein1": vein1, "palm2": palm2, "vein2": vein2,
        }
        counts[category] += 1
        item = (row["severity"], n, row)
        if len(top_rows[category]) < top_k:
            heapq.heappush(top_rows[category], item)
        else:
            heapq.heappushpop(top_rows[category], item)

    rows = [item[2] for items in top_rows.values() for item in items]
    rows.sort(key=lambda row: (row["category"], -row["severity"]))
    write_csv(output_dir / "failure_cases.csv", rows)
    total = len(labels)
    write_csv(output_dir / "failure_summary.csv", [
        {"category": category, "count": count, "fraction": count / total}
        for category, count in sorted(counts.items())
    ])

    shown = [item[2] for items in top_rows.values()
             for item in sorted(items, reverse=True)[:2]]
    if not shown:
        return
    figure, axes = plt.subplots(len(shown), 4, figsize=(8, 2 * len(shown)), squeeze=False)
    for row_axes, row in zip(axes, shown):
        for axis, key in zip(row_axes, ("palm1", "vein1", "palm2", "vein2")):
            axis.imshow(Image.open(row[key]).convert("L"), cmap="gray")
            axis.set_title(key)
            axis.axis("off")
        row_axes[0].set_ylabel(row["category"], fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "failure_examples.png", dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser("Reviewer analyses: confidence and failure cases")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pair_txt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="mobilefacenet", choices=("mobilefacenet", "resnet18"))
    parser.add_argument("--fusion", default="auto", choices=("auto", "stage2", "bottleneck_gate"))
    parser.add_argument("--confidence_npy", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_config.device = str(device)
    checkpoint = safe_torch_load(args.ckpt, device)
    palm_net, feature_dim, _ = build_backbone(args.backbone)
    vein_net, _, _ = build_backbone(args.backbone)
    fusion = build_fusion_model(args.fusion, args.backbone, feature_dim, checkpoint).to(device)
    palm_net.load_state_dict(checkpoint["cnn_palm"])
    vein_net.load_state_dict(checkpoint["cnn_vein"])
    fusion.load_state_dict(checkpoint["fusion"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = PairTxtDataset(args.pair_txt, transform, transform)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    palm_features, vein_features, fusion_features, sample_labels = extract_features(
        (palm_net, vein_net, fusion), loader, device
    )
    scores_by_model = {}
    pair_labels = pair_indices = None
    for name, features in (("palmprint", palm_features), ("palm_vein", vein_features), ("fusion", fusion_features)):
        scores, labels, i, j = build_pair_scores(features, sample_labels)
        scores_by_model[name] = scores
        pair_labels, pair_indices = labels, (i, j)

    metric_rows = []
    for name, scores in scores_by_model.items():
        metric_rows.append({"model": name, **metrics(scores, pair_labels)})
    write_csv(output_dir / "failure_thresholds.csv", metric_rows)
    failure_analysis(dataset, scores_by_model, pair_labels, pair_indices,
                     metric_rows, output_dir, args.top_k)

    confidence_paths = args.confidence_npy
    if confidence_paths is None:
        inferred = Path(args.ckpt).with_name("teacher_confidence_last_epoch.npy")
        confidence_paths = [str(inferred)] if inferred.exists() else []
    if confidence_paths:
        confidence_analysis(confidence_paths, output_dir)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
