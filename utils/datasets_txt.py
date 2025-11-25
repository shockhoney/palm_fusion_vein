import os
import random
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def gen_polyu_list(
    root_dir,
    out_txt="polyu_list.txt",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42):
    random.seed(seed)

    all_pids = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    pid2label = {pid: idx for idx, pid in enumerate(all_pids)}

    lines = []

    for pid in all_pids:
        person_dir = os.path.join(root_dir, pid)
        imgs = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ])
        if not imgs:
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, img_name in enumerate(imgs):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            img_path = os.path.join(person_dir, img_name)
            label = pid2label[pid]

            lines.append(f"{img_path} {label} {split}\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

class TxtImageDataset(Dataset):

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

                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def phase2_list(root_dir: str,
                       train_txt: str,
                       val_txt: str,
                       val_ratio: float = 0.2,
                       seed: int = 42):

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

        ir_path_norm = ir_path.replace("\\", "/")
        vi_path_norm = vi_path.replace("\\", "/")

        pairs.append(f"{ir_path_norm} {vi_path_norm} {label}\n")

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    with open(train_txt, "w", encoding="utf-8") as f:
        f.writelines(train_pairs)
    with open(val_txt, "w", encoding="utf-8") as f:
        f.writelines(val_pairs)

class PairTxtDataset(Dataset):

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
                ir_path, vi_path, label_str = parts[:3]

                palm_path = vi_path.replace("\\", "/")
                vein_path = ir_path.replace("\\", "/")
                label = int(label_str)

                self.samples.append((palm_path, vein_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        palm_path, vein_path, label = self.samples[idx]

        palm_img = Image.open(palm_path).convert('RGB')
        vein_img = Image.open(vein_path).convert('RGB')

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
     gen_polyu_list("C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\PolyU\\NIR",
    out_txt="polyu__NIR_list.txt",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42)
