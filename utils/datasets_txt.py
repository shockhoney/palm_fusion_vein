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

if __name__ == '__main__':
     gen_polyu_list("C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\PolyU\\Red",
    out_txt="polyu__Red_list.txt",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42)
