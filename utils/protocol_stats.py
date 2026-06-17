import argparse
import math
import os
from collections import Counter, defaultdict


SPLITS = {"train", "val", "test"}


def read_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                records.append(parts)
    return records


def pair_counts(labels):
    counts = Counter(labels)
    total = len(labels)
    genuine = sum(math.comb(n, 2) for n in counts.values() if n >= 2)
    all_pairs = math.comb(total, 2) if total >= 2 else 0
    return genuine, all_pairs - genuine


def summarize_image_list(path):
    by_split = defaultdict(list)
    for parts in read_records(path):
        if len(parts) != 3 or parts[2] not in SPLITS:
            continue
        _, label, split = parts
        by_split[split].append(int(label))
    return by_split


def summarize_pair_list(path):
    labels = []
    for parts in read_records(path):
        if len(parts) >= 3:
            labels.append(int(parts[2]))
    return labels


def print_split_summary(name, split, labels):
    counts = Counter(labels)
    genuine, impostor = pair_counts(labels)
    per_id = list(counts.values())
    min_per_id = min(per_id) if per_id else 0
    max_per_id = max(per_id) if per_id else 0
    mean_per_id = (sum(per_id) / len(per_id)) if per_id else 0.0
    print(
        f"{name},{split},{len(counts)},{len(labels)},"
        f"{min_per_id},{mean_per_id:.2f},{max_per_id},{genuine},{impostor}"
    )


def check_overlap(name, split_to_labels):
    split_sets = {split: set(labels) for split, labels in split_to_labels.items()}
    test_labels = split_sets.get("test")
    if not test_labels:
        return
    for split, labels in sorted(split_sets.items()):
        if split == "test":
            continue
        overlap = labels & test_labels
        if overlap:
            print(f"[WARN] {name}: {split}/test share {len(overlap)} identities")


def parse_named_path(value):
    if "=" in value:
        name, path = value.split("=", 1)
        return name, path
    return os.path.splitext(os.path.basename(value))[0], value


def main():
    parser = argparse.ArgumentParser("Summarize identity splits and verification pair counts")
    parser.add_argument("--image-list", action="append", default=[], help="path label split txt file")
    parser.add_argument("--pair-list", action="append", default=[], help="name=path paired txt file")
    args = parser.parse_args()

    print("dataset,split,subjects,samples,min_per_subject,mean_per_subject,max_per_subject,genuine_pairs,impostor_pairs")

    for path in args.image_list:
        name = os.path.splitext(os.path.basename(path))[0]
        split_to_labels = summarize_image_list(path)
        for split in ("train", "val", "test"):
            print_split_summary(name, split, split_to_labels.get(split, []))
        check_overlap(name, split_to_labels)

    grouped_pair_labels = defaultdict(dict)
    for value in args.pair_list:
        name, path = parse_named_path(value)
        if ":" in name:
            dataset, split = name.split(":", 1)
        else:
            dataset, split = name, "all"
        labels = summarize_pair_list(path)
        print_split_summary(dataset, split, labels)
        grouped_pair_labels[dataset][split] = labels

    for dataset, split_to_labels in grouped_pair_labels.items():
        check_overlap(dataset, split_to_labels)


if __name__ == "__main__":
    main()
