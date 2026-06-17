import os
import random
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def gen_polyu_list(root_dir, out_txt="polyu_list.txt", train_ratio=0.8, val_ratio=0.1, seed=42):
    """Generate identity-disjoint train/test lists.

    Identities are first split into train and test. Images from train identities
    are then split into train/val, while all images from test identities remain
    in test.
    """
    random.seed(seed)

    all_pids = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    pid2label = {pid: idx for idx, pid in enumerate(all_pids)}

    shuffled_pids = all_pids[:]
    random.shuffle(shuffled_pids)
    n_train_ids = int(len(shuffled_pids) * train_ratio)
    train_pids = set(shuffled_pids[:n_train_ids])

    lines = []

    for pid in all_pids:
        person_dir = os.path.join(root_dir, pid)
        imgs = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ])
        if not imgs:
            continue

        if pid in train_pids:
            random.shuffle(imgs)
            n_val = int(len(imgs) * val_ratio)
            split_by_img = {
                img_name: ("val" if i < n_val else "train")
                for i, img_name in enumerate(imgs)
            }
        else:
            split_by_img = {img_name: "test" for img_name in imgs}

        for img_name in imgs:
            split = split_by_img[img_name]

            img_path = os.path.relpath(os.path.join(person_dir, img_name)).replace("\\", "/")
            label = pid2label[pid]

            lines.append(f"{img_path} {label} {split}\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

class TxtImageDataset:

    def __init__(self, list_file, split="train", transform=None):
        self.samples = []    
        self.transform = transform

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue

                img_path, label_str, split_str = parts
                if split_str != split:
                    continue

                label = int(label_str)
                img_path = os.path.join(PROJECT_ROOT, img_path)

                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def phase2_list(root_dir, train_txt, val_txt, val_ratio=0.1, seed=42, train_ratio=0.8, test_txt=None):
    """Generate paired lists with identity-disjoint test identities."""

    ir_dir = os.path.join(root_dir, "ir")
    vi_dir = os.path.join(root_dir, "vi")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    missing_vi = []

    for name in sorted(os.listdir(ir_dir)):
        ir_path = os.path.join(ir_dir, name)
        if not os.path.isfile(ir_path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in exts:
            continue

        vi_path = os.path.join(vi_dir, name)
        if not os.path.exists(vi_path):
            missing_vi.append(name)
            continue

        parts = name.split("_")
        person_str = parts[0]          
        label = int(person_str) - 1

        ir_path_norm = os.path.relpath(ir_path).replace("\\", "/")
        vi_path_norm = os.path.relpath(vi_path).replace("\\", "/")

        pairs.append(f"{vi_path_norm} {ir_path_norm} {label}\n")

    by_label = {}
    for line in pairs:
        label = int(line.strip().split()[2])
        by_label.setdefault(label, []).append(line)

    labels = sorted(by_label)
    random.seed(seed)
    random.shuffle(labels)
    n_train_ids = int(len(labels) * train_ratio) if test_txt else len(labels)
    train_labels = set(labels[:n_train_ids])

    train_pairs, val_pairs, test_pairs = [], [], []
    for label in labels:
        label_pairs = by_label[label]
        if label in train_labels:
            random.shuffle(label_pairs)
            n_val = int(len(label_pairs) * val_ratio)
            val_pairs.extend(label_pairs[:n_val])
            train_pairs.extend(label_pairs[n_val:])
        else:
            test_pairs.extend(label_pairs)

    with open(train_txt, "w", encoding="utf-8") as f:
        f.writelines(train_pairs)
    with open(val_txt, "w", encoding="utf-8") as f:
        f.writelines(val_pairs)
    if test_txt:
        with open(test_txt, "w", encoding="utf-8") as f:
            f.writelines(test_pairs)

class PairTxtDataset:

    def __init__(self, list_file, transform_palm=None, transform_vein=None):
        self.samples = []  # 结构: (palm_path, vein_path, label)
        self.transform_palm = transform_palm
        self.transform_vein = transform_vein

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                palm_path, vein_path, label_str = parts[:3]

                palm_path = palm_path.replace("\\", "/")
                vein_path = vein_path.replace("\\", "/")
                label = int(label_str)

                palm_path = os.path.join(PROJECT_ROOT, palm_path)
                vein_path = os.path.join(PROJECT_ROOT, vein_path)

                self.samples.append((palm_path, vein_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        palm_path, vein_path, label = self.samples[idx]

        palm_img = Image.open(palm_path).convert('RGB')
        vein_img = Image.open(vein_path).convert('L')

        if self.transform_palm:
            palm_img = self.transform_palm(palm_img)
        if self.transform_vein:
            vein_img = self.transform_vein(vein_img)

        return palm_img, vein_img,label

# if __name__ == "__main__":
#     root_dir = r"data/CASIA_dataset"
#     train_txt = "casia_phase2_train.txt"
#     val_txt   = "casia_phase2_val.txt"
#     phase2_list(root_dir, train_txt, val_txt, val_ratio=0.2)

if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    gen_polyu_list("data/CASIA/ir",
        out_txt="data_txt/CASIA_palmvein_list.txt",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42)
    gen_polyu_list("data/CASIA/vi",
        out_txt="data_txt/CASIA_palmprint_list.txt",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42)

