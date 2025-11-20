import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm

from models.stage1 import ConvNeXt
from models.stage2 import Stage2Fusion
from utils.loss import ArcFaceCELoss
from utils.custom_datasets import PolyUDataset, CASIADataset

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'outputs/models'
    polyu_nir = 'C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\PolyU\\NIR'
    polyu_red = 'C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\PolyU\\Red'
    casia_vi = 'C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\CASIA_dataset\\vi'
    casia_ir = 'C:\\Users\\admin\\Desktop\\palm_fusion_vein\\data\\CASIA_dataset\\ir'

    p1_epochs, p1_batch, p1_lr = 150, 32, 1e-4
    p1_patience = 20
    p1_train_ratio, p1_val_ratio = 0.8, 0.1

    p2_epochs, p2_batch, p2_lr, p2_enc_lr = 50, 16, 1e-4, 1e-5
    p2_patience = 10

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# class EarlyStopping:
#     def __init__(self, patience=10, min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_value = None
#         self.should_stop = False

#     def __call__(self, current_value, mode='min'):
#         if self.best_value is None:
#             self.best_value = current_value
#             return False

#         improved = (current_value < self.best_value - self.min_delta) if mode == 'min' else \
#                    (current_value > self.best_value + self.min_delta)

#         if improved:
#             self.best_value = current_value
#             self.counter = 0
#         else:
#             self.counter += 1
#             self.should_stop = self.counter >= self.patience

#         return self.should_stop

def get_transforms(strong=True):
    base = [transforms.Resize((224, 224))]
    if strong:
        base += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Reduced from 0.5 to 0.2
        ]
    else:
        base += [
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.05, 0.05))
        ]
    base += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(base)

def create_dataloaders(data_dir, split, batch_size, modality=None):
    if modality:  
        dataset = PolyUDataset(
            data_dir=data_dir,
            split=split,
            transform=get_transforms(strong=(split == 'train')),
            train_ratio=config.p1_train_ratio,
            val_ratio=config.p1_val_ratio,
            seed=42
        )
    else: 
        dataset = CASIADataset(
            palm_dir=config.casia_vi,
            vein_dir=config.casia_ir,
            split=split,
            transform_palm=get_transforms(strong=(split == 'train')),
            transform_vein=get_transforms(strong=(split == 'train')),
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train')
    ), dataset.num_classes

def train_phase1(model, config, writer, model_name, modality):

    data_dir = config.polyu_nir if modality == 'palm' else config.polyu_red
    train_loader, num_classes = create_dataloaders(data_dir, 'train', config.p1_batch, modality)
    val_loader, _ = create_dataloaders(data_dir, 'val', config.p1_batch, modality)

    criterion = ArcFaceCELoss(
        feat_dim=768,
        num_classes=num_classes,
        s=30.0,
        m=0.30,
        warmup_epochs=30
    ).to(config.device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.p1_lr,
        weight_decay=1e-4   
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * config.p1_epochs), int(0.75 * config.p1_epochs)], gamma=0.1)

    # early_stop = EarlyStopping(patience=config.p1_patience)

    best_acc = 0.0 

    for epoch in range(config.p1_epochs):
        model.train()
        criterion.train()
        criterion.set_epoch(epoch)

        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(total=len(train_loader), 
                    desc=f'[{model_name}] Epoch {epoch+1}/{config.p1_epochs}',
                    dynamic_ncols=True)

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            features = model(images, return_spatial=False)
            loss, loss_dict = criterion(features, labels)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()), max_norm=5.0
            )
            optimizer.step()
            train_loss += loss.item()
            train_correct += int(loss_dict['acc'] * labels.size(0))
            train_total += labels.size(0)
            pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total

        scheduler.step()

        model.eval()
        criterion.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_steps = 0
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)

                features = model(images, return_spatial=False)
                loss, loss_dict = criterion(features, labels)

                val_total_loss += loss.item()
                val_correct += int(loss_dict['acc'] * labels.size(0))
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

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save({
                'model': model.state_dict(),          
                'criterion': criterion.state_dict(),  
                'epoch': epoch + 1,                   
                'val_acc': avg_val_acc                   
            }, os.path.join(config.save_dir, f'{model_name}_phase1_best.pth'))

        # if early_stop(-avg_val_acc, mode='min'):
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break
         
    return best_acc

def train_phase2(cnn_palm, cnn_vein, config, writer):

    for model, name in [(cnn_palm, 'cnn_palm'), (cnn_vein, 'cnn_vein')]:
        ckpt_path = os.path.join(config.save_dir, f'{name}_phase1_best.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model'])
        else:
            print(f" {name}not exist ")

    train_loader, num_classes = create_dataloaders(None, 'train', config.p2_batch)
    val_loader, _ = create_dataloaders(None, 'val', config.p2_batch)

    fusion_model = Stage2Fusion(
        in_dim_global_palm=768, in_dim_global_vein=768,
        in_dim_local_palm=768, in_dim_local_vein=768,
        out_dim_local=256, final_l2norm=True,
        ).to(config.device)

    optimizer = torch.optim.Adam([
        {'params': fusion_model.parameters(), 'lr': config.p2_lr},
        {'params': cnn_palm.parameters(), 'lr': config.p2_enc_lr},
        {'params': cnn_vein.parameters(), 'lr': config.p2_enc_lr}
    ], weight_decay=1e-4)  # Reduced from 1e-2 to 1e-4

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    criterion = ArcFaceCELoss(
        feat_dim=512,
        num_classes=num_classes,
        s=30.0,
        m=0.20,
        warmup_epochs=0
    ).to(config.device)

    # early_stop = EarlyStopping(patience=config.p2_patience)

    best_acc = 0.0
 
    for epoch in range(config.p2_epochs):
        cnn_palm.train()
        cnn_vein.train()
        fusion_model.train()

        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(total=len(train_loader),
                    desc=f'[Stage2] Epoch {epoch+1}/{config.p2_epochs}',
                    dynamic_ncols=True)

        for palm_img, vein_img, labels in train_loader:
            palm_img = palm_img.to(config.device)
            vein_img = vein_img.to(config.device)
            labels = labels.to(config.device)

            palm_global = cnn_palm(palm_img, return_spatial=False)
            vein_global = cnn_vein(vein_img, return_spatial=False)
            palm_local = cnn_palm(palm_img, return_spatial=True)
            vein_local = cnn_vein(vein_img, return_spatial=True)

            fused_feat = fusion_model(
                palm_global, vein_global, palm_local, vein_local
            )

            loss,loss_dict = criterion(fused_feat, labels)
            logits = loss_dict['logits']
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_palm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cnn_vein.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)

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
        val_total_loss, val_correct, val_total = 0.0, 0, 0  
        with torch.no_grad():
            val_steps = 0
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device)
                vein_img = vein_img.to(config.device)
                labels = labels.to(config.device)

                palm_global = cnn_palm(palm_img, return_spatial=False)
                vein_global = cnn_vein(vein_img, return_spatial=False)
                palm_local = cnn_palm(palm_img, return_spatial=True)
                vein_local = cnn_vein(vein_img, return_spatial=True)

                fused_feat = fusion_model(palm_global, vein_global, palm_local, vein_local)

                loss,loss_dict = criterion(fused_feat, labels)
                val_total_loss += loss.item()

                _, pred = torch.max(loss_dict['logits'], 1)
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
                'epoch': epoch + 1,
                'cnn_palm': cnn_palm.state_dict(),
                'cnn_vein': cnn_vein.state_dict(),
                'fusion': fusion_model.state_dict(),
                'best_acc': best_acc
            }, os.path.join(config.save_dir, 'stage2_best.pth'))

        # if early_stop(-avg_val_acc, mode='min'):
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_acc

def main():
    writer = SummaryWriter(log_dir='runs')

    cnn_palm = ConvNeXt(in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(config.device)
    cnn_vein = ConvNeXt(in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(config.device)

    skip_stage1 = False  # 设置为 True 跳过 Stage 1

    if not skip_stage1:
        palm_acc = train_phase1(cnn_palm, config, writer, 'cnn_palm', 'palm')
        print(f" Palm(Best Acc: {palm_acc:.2f}%)")

        vein_acc = train_phase1(cnn_vein, config, writer, 'cnn_vein', 'vein')
        print(f" Vein(Best Acc: {vein_acc:.2f}%)")

    best_acc = train_phase2(cnn_palm, cnn_vein, config, writer)
    print(f" best_val_acc: {best_acc:.2f}%")

    writer.close()
    
if __name__ == '__main__':
    main()
