import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PolyUDataset(Dataset):
    def __init__(self, data_dir, split, transform, train_ratio=0.8, val_ratio=0.1, seed=42):
        # 读取所有子目录（person IDs）
        all_pids = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

        # 创建全局标签映射（所有数据集划分共享）
        global_pid_to_label = {pid: idx for idx, pid in enumerate(all_pids)}

        # 按比例划分
        np.random.seed(seed)
        shuffled_pids = np.random.permutation(all_pids)
        num_pids = len(shuffled_pids)
        train_end = int(num_pids * train_ratio)
        val_end = train_end + int(num_pids * val_ratio)

        if split == 'train':
            split_pids = shuffled_pids[:train_end]
        elif split == 'val':
            split_pids = shuffled_pids[train_end:val_end]
        else:
            split_pids = shuffled_pids[val_end:]

        # 构建样本列表（使用全局标签映射）
        self.samples = []
        for pid in split_pids:
            person_dir = os.path.join(data_dir, pid)
            if os.path.exists(person_dir):
                img_files = sorted([f for f in os.listdir(person_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
                label = global_pid_to_label[pid]  # 使用全局标签
                for img_file in img_files:
                    img_path = os.path.join(person_dir, img_file)
                    self.samples.append((img_path, label))

        self.transform = transform
        self.num_classes = len(all_pids)  # 总类别数（不是当前划分的类别数）
        print(f"PolyUDataset ({split}): {len(split_pids)} person IDs, {len(self.samples)} samples, "
              f"{self.num_classes} total classes, label range [{min(s[1] for s in self.samples)}, "
              f"{max(s[1] for s in self.samples)}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


class CASIADataset(Dataset):
    def __init__(self, palm_dir, vein_dir, split, transform_palm, transform_vein,
                 train_ratio=0.8, val_ratio=0.1, seed=42):
        def scan_dir(directory):
            file_dict = {}
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    filepath = os.path.join(directory, filename)
                    person_id = filename.split('_')[0]
                    if person_id not in file_dict:
                        file_dict[person_id] = []
                    file_dict[person_id].append(filepath)
            for pid in file_dict:
                file_dict[pid] = sorted(file_dict[pid])
            return file_dict

        palm_dict = scan_dir(palm_dir)
        vein_dict = scan_dir(vein_dir)

        # 找到同时有掌纹和静脉的 person_id
        all_pids = sorted(set(palm_dict.keys()) & set(vein_dict.keys()))
        if len(all_pids) == 0:
            raise ValueError("No common person IDs found")

        # 创建全局标签映射（所有数据集划分共享）
        global_pid_to_label = {pid: idx for idx, pid in enumerate(all_pids)}

        # 按身份为单位划分
        np.random.seed(seed)
        shuffled_pids = np.random.permutation(all_pids)
        num_pids = len(shuffled_pids)
        train_end = int(num_pids * train_ratio)
        val_end = train_end + int(num_pids * val_ratio)

        if split == 'train':
            split_pids = shuffled_pids[:train_end]
        elif split == 'val':
            split_pids = shuffled_pids[train_end:val_end]
        else:
            split_pids = shuffled_pids[val_end:]

        self.num_classes = len(all_pids)  # 总类别数（不是当前划分的类别数）

        # 构造成对样本列表（使用全局标签映射）
        self.palm_imgs, self.vein_imgs, self.labels = [], [], []

        for pid in split_pids:
            p_list = palm_dict[pid]
            v_list = vein_dict[pid]
            k = min(len(p_list), len(v_list))
            label = global_pid_to_label[pid]  # 使用全局标签
            for i in range(k):
                self.palm_imgs.append(p_list[i])
                self.vein_imgs.append(v_list[i])
                self.labels.append(label)

        self.transform_palm = transform_palm
        self.transform_vein = transform_vein
        print(f"CASIADataset ({split}): {len(split_pids)} person IDs, {len(self.palm_imgs)} samples, "
              f"{self.num_classes} total classes, label range [{min(self.labels)}, {max(self.labels)}]")

    def __len__(self):
        return len(self.palm_imgs)

    def __getitem__(self, idx):
        palm_path = self.palm_imgs[idx]
        vein_path = self.vein_imgs[idx]
        label = self.labels[idx]

        palm_img = Image.open(palm_path).convert('L')
        vein_img = Image.open(vein_path).convert('L')

        if self.transform_palm is not None:
            palm_img = self.transform_palm(palm_img)
        if self.transform_vein is not None:
            vein_img = self.transform_vein(vein_img)

        return palm_img, vein_img, label
