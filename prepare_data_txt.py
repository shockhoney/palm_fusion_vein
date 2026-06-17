from collections import defaultdict
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "data_txt"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def rel(path):
    return path.relative_to(ROOT).as_posix()


def collect(root, session):
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTS:
            identity = path.parent.name
            files[(identity, session, path.name)] = path
    return files


def val_count(n):
    if n < 4:
        return 0
    return min(n - 1, max(2, round(n * VAL_RATIO)))


def build_dataset(name, palm_roots, vein_roots, single_prefix, pair_prefix, group_fn=None):
    palm = {}
    vein = {}
    for session, root in palm_roots:
        palm.update(collect(root, session))
    for session, root in vein_roots:
        vein.update(collect(root, session))

    keys = sorted(set(palm) & set(vein))
    identities = sorted({key[0] for key in keys})
    label_by_id = {identity: i for i, identity in enumerate(identities)}
    group_fn = group_fn or (lambda identity: identity)

    groups = sorted({group_fn(identity) for identity in identities})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    train_groups = set(groups[: int(len(groups) * TRAIN_RATIO)])

    by_identity = defaultdict(list)
    for key in keys:
        by_identity[key[0]].append(key)

    split_by_key = {}
    for identity, identity_keys in by_identity.items():
        identity_keys = sorted(identity_keys)
        if group_fn(identity) not in train_groups:
            for key in identity_keys:
                split_by_key[key] = "test"
            continue

        rng.shuffle(identity_keys)
        n_val = val_count(len(identity_keys))
        for key in identity_keys[:n_val]:
            split_by_key[key] = "val"
        for key in identity_keys[n_val:]:
            split_by_key[key] = "train"

    palm_lines, vein_lines = [], []
    pair_lines = {"train": [], "val": [], "test": []}
    for key in keys:
        identity = key[0]
        label = label_by_id[identity]
        split = split_by_key[key]
        palm_path = rel(palm[key])
        vein_path = rel(vein[key])
        palm_lines.append(f"{palm_path} {label} {split}\n")
        vein_lines.append(f"{vein_path} {label} {split}\n")
        pair_lines[split].append(f"{palm_path} {vein_path} {label}\n")

    OUT.mkdir(exist_ok=True)
    (OUT / f"{single_prefix}_palmprint_list.txt").write_text("".join(palm_lines), encoding="utf-8")
    (OUT / f"{single_prefix}_palmvein_list.txt").write_text("".join(vein_lines), encoding="utf-8")
    for split, lines in pair_lines.items():
        (OUT / f"{pair_prefix}_phase2_{split}.txt").write_text("".join(lines), encoding="utf-8")

    print(
        f"{name}: identities={len(identities)}, pairs={len(keys)}, "
        f"train={len(pair_lines['train'])}, val={len(pair_lines['val'])}, test={len(pair_lines['test'])}"
    )


def main():
    build_dataset(
        "CASIA",
        [("", DATA / "CASIA" / "vi")],
        [("", DATA / "CASIA" / "ir")],
        "CASIA",
        "CASIA",
        group_fn=lambda identity: identity.split("_")[0],
    )
    build_dataset(
        "CUMT",
        [("", DATA / "CUMT" / "palmprint")],
        [("", DATA / "CUMT" / "palmvein")],
        "CUMT",
        "CUMT",
    )
    build_dataset(
        "PolyU",
        [("", DATA / "PolyU" / "Red")],
        [("", DATA / "PolyU" / "NIR")],
        "PolyU",
        "polyu",
    )
    build_dataset(
        "tongji",
        [("session1", DATA / "tongji" / "palm_session1"), ("session2", DATA / "tongji" / "palm_session2")],
        [("session1", DATA / "tongji" / "vein_session1"), ("session2", DATA / "tongji" / "vein_session2")],
        "tongji",
        "tongji",
    )


if __name__ == "__main__":
    main()
