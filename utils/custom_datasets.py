import os
import numpy as np
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


class PolyUDataset(Dataset):

    def __init__(self, data_dir, split, transform, train_ratio=0.8, val_ratio=0.1, seed=42):
        # Get person IDs from directory structure
        all_pids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        split_pids = split_person_ids(all_pids, split, train_ratio, val_ratio, seed)

        # Map PIDs to labels
        pid_to_label = {pid: idx for idx, pid in enumerate(sorted(split_pids))}

        # Collect all image samples
        self.samples = []
        for pid in split_pids:
            person_dir = os.path.join(data_dir, pid)
            images = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(IMAGE_EXTS)])
            self.samples.extend([(os.path.join(person_dir, img), pid_to_label[pid]) for img in images])

        self.transform = transform
        self.num_classes = len(split_pids)
        print(f"PolyUDataset ({split}): {len(split_pids)} PIDs, {len(self.samples)} samples, "
              f"labels [{min(s[1] for s in self.samples)}-{max(s[1] for s in self.samples)}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        return self.transform(img) if self.transform else img, label


class CASIADataset(Dataset):
    
    def __init__(self, palm_dir, vein_dir, split, transform_palm, transform_vein,
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        # Scan both directories for images
        palm_dict = self._scan_dir(palm_dir)
        vein_dict = self._scan_dir(vein_dir)

        # Find common person IDs
        all_pids = sorted(set(palm_dict.keys()) & set(vein_dict.keys()))
        if not all_pids:
            raise ValueError("No common person IDs found")

        split_pids = split_person_ids(all_pids, split, train_ratio, val_ratio, seed)
        pid_to_label = {pid: idx for idx, pid in enumerate(sorted(split_pids))}

        # Pair palm and vein images
        self.palm_imgs, self.vein_imgs, self.labels = [], [], []
        for pid in split_pids:
            n_pairs = min(len(palm_dict[pid]), len(vein_dict[pid]))
            for i in range(n_pairs):
                self.palm_imgs.append(palm_dict[pid][i])
                self.vein_imgs.append(vein_dict[pid][i])
                self.labels.append(pid_to_label[pid])

        self.transform_palm = transform_palm
        self.transform_vein = transform_vein
        self.num_classes = len(split_pids)
        print(f"CASIADataset ({split}): {len(split_pids)} PIDs, {len(self.labels)} samples, "
              f"labels [{min(self.labels)}-{max(self.labels)}]")

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
