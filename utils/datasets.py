import os
import numpy as np
import zlib
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = ('.jpg', '.png', '.jpeg', '.bmp')

def split_person_ids(all_pids, split, train_ratio=0.8, val_ratio=0.1, seed=42):

    np.random.seed(seed)
    shuffled = np.random.permutation(sorted(all_pids))
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    ranges = {'train': (0, train_end), 'val': (train_end, val_end), 'test': (val_end, n)}
    start, end = ranges.get(split, (val_end, n))
    return shuffled[start:end]


def split_indices_by_ratio(num_items, split, train_ratio=0.8, val_ratio=0.1, seed=42):

    rng = np.random.RandomState(seed)
    order = rng.permutation(num_items)
    train_end = int(num_items * train_ratio)
    val_end = train_end + int(num_items * val_ratio)
    ranges = {'train': (0, train_end), 'val': (train_end, val_end), 'test': (val_end, num_items)}
    start, end = ranges.get(split, (val_end, num_items))
    return order[start:end]

def _seed_from_pid(pid: str, base_seed: int) -> int:
    return base_seed + (zlib.crc32(pid.encode('utf-8')) % 10000)


class PolyUDataset(Dataset):

    def __init__(self, data_dir, split, transform, train_ratio=0.8, val_ratio=0.1, seed=42):

        self.all_pids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.pid_to_label = {pid: idx for idx, pid in enumerate(self.all_pids)}
        self.num_classes = len(self.all_pids)

        self.samples = []
        for pid in self.all_pids:
            person_dir = os.path.join(data_dir, pid)
            images = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(IMAGE_EXTS)])
            if not images:
                continue

            pid_seed = _seed_from_pid(pid, seed)
            idxs = split_indices_by_ratio(len(images), split, train_ratio, val_ratio, pid_seed)
            for idx in idxs:
                self.samples.append((os.path.join(person_dir, images[idx]), self.pid_to_label[pid]))

        self.transform = transform
        label_range = (min(s[1] for s in self.samples), max(s[1] for s in self.samples)) if self.samples else (0, 0)
        print(f"PolyUDataset ({split}): {len(self.all_pids)} PIDs, {len(self.samples)} samples, "
              f"labels [{label_range[0]}-{label_range[1]}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        return self.transform(img) if self.transform else img, label


class CASIADataset(Dataset):
    
    def __init__(self, palm_dir, vein_dir, split, transform_palm, transform_vein,
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        palm_dict = self._scan_dir(palm_dir)
        vein_dict = self._scan_dir(vein_dir)

        all_pids = sorted(set(palm_dict.keys()) & set(vein_dict.keys()))
        if not all_pids:
            raise ValueError("No common person IDs found")

        pid_to_label = {pid: idx for idx, pid in enumerate(sorted(all_pids))}

        self.palm_imgs, self.vein_imgs, self.labels = [], [], []
        for pid in all_pids:
            n_pairs = min(len(palm_dict[pid]), len(vein_dict[pid]))
            if n_pairs == 0:
                continue
            pid_seed = _seed_from_pid(pid, seed)
            pair_order = split_indices_by_ratio(n_pairs, split, train_ratio, val_ratio, pid_seed)

            for idx in pair_order:
                self.palm_imgs.append(palm_dict[pid][idx])
                self.vein_imgs.append(vein_dict[pid][idx])
                self.labels.append(pid_to_label[pid])

        self.transform_palm = transform_palm
        self.transform_vein = transform_vein
        self.num_classes = len(all_pids)
        label_range = (min(self.labels), max(self.labels)) if self.labels else (0, 0)
        print(f"CASIADataset ({split}): {len(all_pids)} PIDs, {len(self.labels)} samples, "
              f"labels [{label_range[0]}-{label_range[1]}]")

    @staticmethod
    def _scan_dir(directory):
        """Scan directory and group images by person ID."""
        file_dict = {}
        for filename in os.listdir(directory):
            if filename.lower().endswith(IMAGE_EXTS):
                person_id = filename.split('_')[0]
                file_dict.setdefault(person_id, []).append(os.path.join(directory, filename))
        return {pid: sorted(paths) for pid, paths in file_dict.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        palm_img = Image.open(self.palm_imgs[idx]).convert('L')
        vein_img = Image.open(self.vein_imgs[idx]).convert('L')

        if self.transform_palm:
            palm_img = self.transform_palm(palm_img)
        if self.transform_vein:
            vein_img = self.transform_vein(vein_img)

        return palm_img, vein_img, self.labels[idx]
