import argparse
import csv
from pathlib import Path

import numpy as np


def best_row(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["eer"]))


def main():
    parser = argparse.ArgumentParser("Summarize val_metrics.csv files")
    parser.add_argument("--root", default="outputs/sweeps")
    args = parser.parse_args()

    groups = {}
    for path in Path(args.root).rglob("val_metrics.csv"):
        row = best_row(path)
        if row is None:
            continue
        name = path.parent.name
        group = name.rsplit("_seed", 1)[0]
        groups.setdefault(group, []).append(row)

    print("group,n,eer_mean,eer_std,tar_1e-5_mean,tar_1e-5_std,tar_1e-4_mean,tar_1e-4_std")
    for group, rows in sorted(groups.items()):
        eer = np.array([float(r["eer"]) for r in rows])
        tar5 = np.array([float(r["tar_1e-05"]) for r in rows])
        tar4 = np.array([float(r["tar_1e-04"]) for r in rows])
        print(
            f"{group},{len(rows)},"
            f"{eer.mean():.6f},{eer.std(ddof=1) if len(rows) > 1 else 0:.6f},"
            f"{tar5.mean():.6f},{tar5.std(ddof=1) if len(rows) > 1 else 0:.6f},"
            f"{tar4.mean():.6f},{tar4.std(ddof=1) if len(rows) > 1 else 0:.6f}"
        )


if __name__ == "__main__":
    main()
