import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torch


class SimpleDataset(Dataset):
    """Simple dataset for flat directory structure (Stage-1 training)

    Expects filenames like: 001_l_01.jpg, 001_r_02.jpg
    Where 001 is the person ID
    """
    def __init__(self, palm_dir, vein_dir, transform=None):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform
        self.samples = []
        self.id_to_label = {}

        # Collect all images from both directories
        all_files = []

        # Palm images
        if os.path.exists(palm_dir):
            for f in os.listdir(palm_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    person_id = f.split('_')[0]
                    all_files.append((os.path.join(palm_dir, f), person_id))

        # Vein images
        if os.path.exists(vein_dir):
            for f in os.listdir(vein_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    person_id = f.split('_')[0]
                    all_files.append((os.path.join(vein_dir, f), person_id))

        # Create ID to label mapping
        unique_ids = sorted(set(person_id for _, person_id in all_files))
        self.id_to_label = {pid: idx for idx, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)

        # Create samples list
        for path, person_id in all_files:
            label = self.id_to_label[person_id]
            self.samples.append((path, label))

        print(f"SimpleDataset: {self.num_classes} classes, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label


class ContrastDataset(Dataset):   
    def __init__(self, palm_dir, vein_dir, transform=None):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform

        self.id_dict = {}

        for f in os.listdir(palm_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                pid = f.split('_')[0]
                if pid not in self.id_dict:
                    self.id_dict[pid] = {'palm': [], 'vein': []}
                self.id_dict[pid]['palm'].append(f)
        
        for f in os.listdir(vein_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                pid = f.split('_')[0]
                if pid not in self.id_dict:
                    self.id_dict[pid] = {'palm': [], 'vein': []}
                self.id_dict[pid]['vein'].append(f)
        
        self.all_ids = sorted(self.id_dict.keys())
        self.pid_to_label = {pid: i for i, pid in enumerate(self.all_ids)}

        self.samples = []
        for pid in self.all_ids:
            for f in self.id_dict[pid]['palm']:
                self.samples.append(('palm', pid, f))
            for f in self.id_dict[pid]['vein']:
                self.samples.append(('vein', pid, f))
        
        print(f"dataset：{len(self.all_ids)}persons，{len(self.samples)}samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        modality, person_id, anchor_file = self.samples[idx]
        anchor_path = os.path.join(self.palm_dir if modality == 'palm' else self.vein_dir, anchor_file)
        anchor_img = Image.open(anchor_path).convert('L')
        
        pos_candidates = []
        for f in self.id_dict[person_id]['palm']:
            if not (modality == 'palm' and f == anchor_file):
                pos_candidates.append(('palm', f))
        for f in self.id_dict[person_id]['vein']:
            if not (modality == 'vein' and f == anchor_file):
                pos_candidates.append(('vein', f))
        
        if pos_candidates:
            pos_mod, pos_file = random.choice(pos_candidates)
        else:
            pos_mod, pos_file = modality, anchor_file
        
        pos_path = os.path.join(self.palm_dir if pos_mod == 'palm' else self.vein_dir, pos_file)
        positive_img = Image.open(pos_path).convert('L')
        
        neg_candidates = [pid for pid in self.all_ids if pid != person_id and (self.id_dict[pid]['palm'] + self.id_dict[pid]['vein'])]
        if not neg_candidates:
            raise RuntimeError('No negative candidates found in dataset')
        neg_id = random.choice(neg_candidates)
        neg_files = self.id_dict[neg_id]['palm'] + self.id_dict[neg_id]['vein']
        neg_file = random.choice(neg_files)
        
        if neg_file in self.id_dict[neg_id]['palm']:
            neg_path = os.path.join(self.palm_dir, neg_file)
        else:
            neg_path = os.path.join(self.vein_dir, neg_file)
        negative_img = Image.open(neg_path).convert('L')
        
        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            except Exception as e:
                raise RuntimeError(f"Error applying transform: {e}")

        label = self.pid_to_label.get(person_id, 0)
        return anchor_img, positive_img, negative_img, label
    

class PairDataset(Dataset):
    def __init__(self, palm_dir, vein_dir, transform=None, split='train'):
        self.palm_dir = palm_dir
        self.vein_dir = vein_dir
        self.transform = transform

        # 读取所有文件
        self.palm_imgs, self.vein_imgs, self.raw_labels = self.load_files(split)

        # 自动生成 label 映射
        unique_labels = sorted(set(self.raw_labels))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        # 映射 labels
        self.labels = torch.tensor([self.label_map[l] for l in self.raw_labels], dtype=torch.long)

    def load_files(self, split):
        palm_imgs, vein_imgs, labels = [], [], []

        # 假设每个类别是一个子文件夹
        for idx, cls in enumerate(sorted(os.listdir(self.palm_dir))):
            cls_palm_dir = os.path.join(self.palm_dir, cls)
            cls_vein_dir = os.path.join(self.vein_dir, cls)
            if not os.path.isdir(cls_palm_dir) or not os.path.isdir(cls_vein_dir):
                continue

            imgs_palm = sorted(os.listdir(cls_palm_dir))
            imgs_vein = sorted(os.listdir(cls_vein_dir))
            num_samples = min(len(imgs_palm), len(imgs_vein))

            #  修复：清理冗余的三元表达式
            split_idx = int(num_samples * 0.8)
            start, end = (0, split_idx) if split == 'train' else (split_idx, num_samples)

            for i in range(start, end):
                palm_imgs.append(os.path.join(cls_palm_dir, imgs_palm[i]))
                vein_imgs.append(os.path.join(cls_vein_dir, imgs_vein[i]))
                labels.append(idx)  # 用 idx 作为原始 label

        return palm_imgs, vein_imgs, labels

    def __len__(self):
        return len(self.palm_imgs)

    def __getitem__(self, idx):
        palm = Image.open(self.palm_imgs[idx]).convert('L')
        vein = Image.open(self.vein_imgs[idx]).convert('L')
        label = self.labels[idx]

        if self.transform is not None:
            palm = self.transform(palm)
            vein = self.transform(vein)

        return palm, vein, label
