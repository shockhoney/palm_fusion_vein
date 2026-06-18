import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from models.stage1_mobileFacenet import MobileFaceNet
from models.resnet18_encoder import ResNet18Encoder
from models.stage2 import Stage2Fusion

from utils.head import Arcface_Head 
from utils.datasets_txt import TxtImageDataset, PairTxtDataset

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    save_dir = 'outputs/models'
    backbone = 'resnet18'  
    pretrained_path = 'pretrain/resnet18_imagenet1k_v1.pth'
    input_size = 224
    num_workers = 8
    seed = 42

    list_file_palm = 'data_txt/CASIA_palmprint_list.txt'
    list_file_vein = 'data_txt/CASIA_palmvein_list.txt'
    phase2_train = 'data_txt/CASIA_phase2_train.txt'
    phase2_val = 'data_txt/CASIA_phase2_val.txt'

    p1_epochs, p1_batch, p1_lr = 200, 8, 1e-2
    p1_patience = 100
    p2_epochs, p2_batch, p2_lr, p2_enc_lr = 200, 8, 1e-3, 1e-4
    p2_patience = 100

config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def build_backbone(name):
    name = name.lower()
    if name == 'resnet18':
        model = ResNet18Encoder(
            input_channel=3,
            input_size=config.input_size,
            pretrained_path=config.pretrained_path,
        ).to(config.device)
        feat_dim = model.out_dim
        local_dim = model.local_dim
    elif name == 'mobilefacenet':
        model = MobileFaceNet(input_channel=3, input_size=config.input_size).to(config.device)
        feat_dim = model.out_dim
        local_dim = model.local_dim
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return model, feat_dim, local_dim

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value, mode='min'):
        if self.best_value is None:
            self.best_value = current_value
            return False

        improved = (current_value < self.best_value - self.min_delta) if mode == 'min' else \
                   (current_value > self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience

        return self.should_stop

def get_transforms(img_size, strong=True):
    base = [transforms.Resize((img_size, img_size))]
    if strong:
        base += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  
        ]
    else:
        base += [
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.05, 0.05))
        ]
    base += [transforms.Grayscale(num_output_channels=3),transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
]
    return transforms.Compose(base)

def create_dataloaders_from_txt(list_file, batch_size):
    train_tf = get_transforms(config.input_size, strong=True)
    val_tf   = get_transforms(config.input_size, strong=False)

    train_dataset = TxtImageDataset(list_file=list_file, split="train", transform=train_tf)
    val_dataset   = TxtImageDataset(list_file=list_file, split="val",   transform=val_tf)

    labels = [label for _, label in train_dataset.samples]
    num_classes = max(labels) + 1 if labels else 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.num_workers, worker_init_fn=seed_worker,
        generator=make_generator(config.seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.num_workers, worker_init_fn=seed_worker,
        generator=make_generator(config.seed + 1)
    )

    return train_loader, val_loader, num_classes

def create_phase2_dataloaders(train_list, val_list, batch_size):
    train_tf = get_transforms(config.input_size, strong=True)
    val_tf   = get_transforms(config.input_size, strong=False)

    train_dataset = PairTxtDataset(list_file=train_list,transform_palm=train_tf,transform_vein=train_tf)
    val_dataset = PairTxtDataset(list_file=val_list,transform_palm=val_tf,transform_vein=val_tf)

    labels = [label for _, _, label in train_dataset.samples]
    num_classes = max(labels) + 1 if labels else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(config.seed + 2)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(config.seed + 3)
    )
    return train_loader, val_loader, num_classes


def train_phase1(model, config, writer, model_name, feat_dim):

    name_low = model_name.lower()
    if 'palm' in name_low:
        list_file = config.list_file_palm
    elif 'vein' in name_low:
        list_file = config.list_file_vein

    train_loader, val_loader, num_classes = create_dataloaders_from_txt(list_file, config.p1_batch)

    classifier = Arcface_Head(embedding_size=feat_dim,num_classes=num_classes,s=32.0,m=0.25).to(config.device)

    # classifier = nn.Linear(feat_dim, num_classes).to(config.device)
    ce_loss = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(
    #     list(model.parameters()) + list(classifier.parameters()),
    #     lr=config.p1_lr,weight_decay=1e-4)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.p1_lr, momentum=0.9, weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(0.5 * config.p1_epochs),int(0.75 * config.p1_epochs)], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p1_epochs)
    early_stop = EarlyStopping(patience=config.p1_patience)
    best_acc = 0.0 

    for epoch in range(config.p1_epochs):
        model.train()
        classifier.train()

        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(total=len(train_loader), 
                    desc=f'[{model_name}] Epoch {epoch+1}/{config.p1_epochs}',
                    dynamic_ncols=True)

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            features = model(images, return_spatial=False)
            logits = classifier(features, labels)
            loss = ce_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                 list(model.parameters()) + list(classifier.parameters()), max_norm=5.0
             )

            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total

        model.eval()
        classifier.eval()

        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_steps = 0
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)

                features = model(images, return_spatial=False)
                logits = classifier(features, labels)

                loss = ce_loss(logits, labels)
                val_total_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_steps += 1

            avg_val_loss = val_total_loss / val_steps
            avg_val_acc = 100. * val_correct / val_total

            pbar.set_postfix({
                'TrLoss': f"{avg_train_loss:.4f}",
                'TrAcc': f"{avg_train_acc:.2f}%",
                'VaLoss': f"{avg_val_loss:.4f}",
                'VaAcc': f"{avg_val_acc:.2f}%"
                 })
        pbar.close()

        if writer:
            writer.add_scalar(f'Phase1_{model_name}/TrainLoss', avg_train_loss, epoch)
            writer.add_scalar(f'Phase1_{model_name}/TrainAcc', avg_train_acc, epoch)
            writer.add_scalar(f'Phase1_{model_name}/ValLoss', avg_val_loss, epoch)
            writer.add_scalar(f'Phase1_{model_name}/ValAcc', avg_val_acc, epoch)

        scheduler.step()

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'backbone': config.backbone,
                'pretrained_path': config.pretrained_path,
                'model': model.state_dict(),          
                 'classifier': classifier.state_dict()                
            }, os.path.join(config.save_dir, f'{model_name}_phase1_best_demo.pth'))

        if early_stop(-avg_val_acc, mode='min'):
            print(f"Early stopping at epoch {epoch+1}")
            break
         
    return best_acc

def train_phase2(cnn_palm, cnn_vein, config, writer, feat_dim, local_dim):

    for model, name in [(cnn_palm, 'cnn_palm'), (cnn_vein, 'cnn_vein')]:
        ckpt_path = os.path.join(config.save_dir, f'{name}_phase1_best_demo.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model'])
        else:
            print(f" {name}not exist ")

    train_loader, val_loader, num_classes = create_phase2_dataloaders( config.phase2_train,config.phase2_val,config.p2_batch)

    fusion_model = Stage2Fusion(in_dim_global=feat_dim,out_dim_final=512,final_l2norm=True).to(config.device)

    classifier = Arcface_Head(
        embedding_size=512,
        num_classes=num_classes,
        s=30.0,
        m=0.20,
    ).to(config.device)
    # classifier = nn.Linear(2*feat_dim, num_classes).to(config.device)
    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': fusion_model.parameters(), 'lr': config.p2_lr},
        {'params': cnn_palm.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein.parameters(), 'lr': config.p2_enc_lr},
        {'params': classifier.parameters(), 'lr': config.p2_lr}], weight_decay=1e-4)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    early_stop = EarlyStopping(patience=config.p2_patience)
    best_acc = 0.0
 
    for epoch in range(config.p2_epochs):
        cnn_palm.train()
        cnn_vein.train()
        fusion_model.train()

        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[Fusion] Epoch {epoch+1}/{config.p2_epochs}',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device)
            vein_img = vein_img.to(config.device)
            labels = labels.to(config.device)

            F_palm = cnn_palm(palm_img, return_spatial=False)
            F_vein = cnn_vein(vein_img, return_spatial=False)

            fused_feat = fusion_model(F_palm, F_vein)

            logits = classifier(fused_feat, labels)
            loss = ce_loss(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_palm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(),   1.0)

            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total

        scheduler.step()

        cnn_palm.eval()
        cnn_vein.eval()
        fusion_model.eval()
        classifier.eval()

        val_total_loss, val_correct, val_total = 0.0, 0, 0  
        with torch.no_grad():
            val_steps = 0
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device)
                vein_img = vein_img.to(config.device)
                labels = labels.to(config.device)

                palm_global = cnn_palm(palm_img, return_spatial=False)
                vein_global = cnn_vein(vein_img, return_spatial=False)

                fused_feat = fusion_model(palm_global, vein_global)

                logits = classifier(fused_feat, labels)
                loss = ce_loss(logits, labels)
                val_total_loss += loss.item()

                _, pred = torch.max(logits, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                val_steps += 1

            avg_val_loss = val_total_loss / val_steps
            avg_val_acc = 100. * val_correct / val_total

            pbar.set_postfix({
                'TrLoss': f"{avg_train_loss:.4f}",
                'TrAcc': f"{avg_train_acc:.2f}%",
                'VaLoss': f"{avg_val_loss:.4f}",
                'VaAcc': f"{avg_val_acc:.2f}%"
            })
        pbar.close()

        if writer:
            writer.add_scalar('Phase2/TrainLoss', avg_train_loss, epoch)
            writer.add_scalar('Phase2/TrainAcc', avg_train_acc, epoch)
            writer.add_scalar('Phase2/ValLoss', avg_val_loss, epoch)
            writer.add_scalar('Phase2/ValAcc', avg_val_acc, epoch)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'backbone': config.backbone,
                'pretrained_path': config.pretrained_path,
                'cnn_palm': cnn_palm.state_dict(),
                'cnn_vein': cnn_vein.state_dict(),
                'fusion': fusion_model.state_dict(),
                'classifier': classifier.state_dict()
            }, os.path.join(config.save_dir, 'stage2_best.pth'))

        if early_stop(-avg_val_acc, mode='min'):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_acc

def main():
    parser = argparse.ArgumentParser("Train teacher fusion model")
    parser.add_argument("--backbone", type=str, default=config.backbone)
    parser.add_argument("--pretrained_path", type=str, default=config.pretrained_path)
    parser.add_argument("--list_file_palm", type=str, default=config.list_file_palm)
    parser.add_argument("--list_file_vein", type=str, default=config.list_file_vein)
    parser.add_argument("--phase2_train", type=str, default=config.phase2_train)
    parser.add_argument("--phase2_val", type=str, default=config.phase2_val)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--save_dir", type=str, default=config.save_dir)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--skip_stage1", action="store_true")
    args = parser.parse_args()

    config.backbone = args.backbone
    config.pretrained_path = args.pretrained_path
    config.list_file_palm = args.list_file_palm
    config.list_file_vein = args.list_file_vein
    config.phase2_train = args.phase2_train
    config.phase2_val = args.phase2_val
    config.seed = args.seed
    run_name = args.run_name
    config.save_dir = os.path.join(args.save_dir, run_name) if run_name else args.save_dir
    os.makedirs(config.save_dir, exist_ok=True)
    set_seed(config.seed)

    log_dir = os.path.join('runs', run_name or f"seed_{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)

    cnn_palm, feat_dim, local_dim = build_backbone(config.backbone)
    cnn_vein, _, _ = build_backbone(config.backbone)

    skip_stage1 = args.skip_stage1

    if not skip_stage1:

        palm_acc = train_phase1(cnn_palm, config, writer, 'cnn_palm', feat_dim)
        print(f" Palm(Best Acc: {palm_acc:.2f}%)")
        vein_acc = train_phase1(cnn_vein, config, writer, 'cnn_vein', feat_dim)
        print(f" Vein(Best Acc: {vein_acc:.2f}%)")

    best_acc = train_phase2(cnn_palm, cnn_vein, config, writer, feat_dim, local_dim)
    print(f" best_val_acc: {best_acc:.2f}%")

    writer.close()
    
if __name__ == '__main__':
    main()
