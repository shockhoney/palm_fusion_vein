import argparse
import subprocess
import sys


DATASETS = {
    "PolyU": ("data_txt/polyu_phase2_train.txt", "data_txt/polyu_phase2_val.txt"),
    "CASIA": ("data_txt/CASIA_phase2_train.txt", "data_txt/CASIA_phase2_val.txt"),
    "CUMT": ("data_txt/CUMT_phase2_train.txt", "data_txt/CUMT_phase2_val.txt"),
    "tongji": ("data_txt/tongji_phase2_train.txt", "data_txt/tongji_phase2_val.txt"),
}


def main():
    parser = argparse.ArgumentParser("Run student distillation sweeps")
    parser.add_argument("--dataset", choices=DATASETS, default="PolyU")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--save_dir", default="outputs/sweeps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_list, val_list = DATASETS[args.dataset]
    for batch_size in args.batch_sizes:
        for seed in args.seeds:
            run_name = f"{args.dataset}_bs{batch_size}_seed{seed}"
            cmd = [
                sys.executable, "train_s_vkd.py",
                "--train_list", train_list,
                "--val_list", val_list,
                "--teacher_ckpt", args.teacher_ckpt,
                "--save_dir", args.save_dir,
                "--run_name", run_name,
                "--seed", str(seed),
                "--batch_size", str(batch_size),
                "--epochs", str(args.epochs),
            ]
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
