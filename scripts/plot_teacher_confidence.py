import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser("Plot teacher confidence distribution")
    parser.add_argument("--npy", required=True)
    parser.add_argument("--out", default="teacher_confidence_hist.png")
    args = parser.parse_args()

    conf = np.load(args.npy)
    print(f"count,{conf.size}")
    print(f"mean,{conf.mean():.6f}")
    print(f"std,{conf.std():.6f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"p{p},{np.percentile(conf, p):.6f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.hist(conf, bins=40, color="#4c72b0", edgecolor="white")
    plt.xlabel("Teacher confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
