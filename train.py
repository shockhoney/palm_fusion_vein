import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from utils.loss import get_stage1_loss, get_stage2_loss
from utils.dataset import SimpleDataset, PairDataset
from utils.metrics import compute_eer, roc_auc, tar_at_far
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from models.stage1 import ConvNeXt
from models.stage2 import Stage2FusionCA 

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'outputs/models' 
    palm_dir1, vein_dir1 = 'C:/Users\ganji/Desktop/pp_pv_fusion/data/CASIA_dataset/vi', 'C:/Users\ganji/Desktop/pp_pv_fusion/data/CASIA_dataset/ir'
    palm_dir2, vein_dir2 = 'C:/Users/ganji/Desktop/pp_pv_fusion/data/PolyU/NIR', 'C:/Users/ganji/Desktop/pp_pv_fusion/data/PolyU/Red'
    p1_epochs, p1_batch, p1_lr = 150, 8, 1e-4
    p1_patience = 140 
    p2_epochs, p2_batch, p2_lr, p2_enc_lr = 100, 8, 1e-4, 1e-5 
    p2_patience = 15 
    num_classes, train_ratio = 100, 0.8

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

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
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop

def get_transforms(strong=True):
    # 修改为 224x224 以匹配模型输入
    base = [transforms.Resize((224, 224))]
    if strong:
        base += [transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3)]
    else:
        base += [transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.05, 0.05))]

    base += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(base)

def train_phase1_cnn(cnn_model, config, writer=None, model_name='cnn_palm'):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = SimpleDataset(config.palm_dir1, config.vein_dir1, get_transforms(strong=True))
    loader = DataLoader(dataset, config.p1_batch, shuffle=True, num_workers=4, drop_last=True)

    actual_num_classes = dataset.num_classes

    criterion = get_stage1_loss(
        feat_dim=768,
        num_classes=actual_num_classes,
        s=30.0,
        m=0.35,           # 角度间隔
        lambda_center=0.008,  #
        lambda_margin=0.001,
        margin=2.0
    ).to(config.device)

    optimizer = torch.optim.Adam(
        list(cnn_model.parameters()) + list(criterion.parameters()),
        lr=config.p1_lr,
        weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p1_epochs)
    early_stop = EarlyStopping(patience=config.p1_patience, min_delta=0.001)

    best_loss = float('inf')

    for epoch in range(config.p1_epochs):
        cnn_model.train()
        criterion.train()
        criterion.set_epoch(epoch)  # 设置当前epoch用于warmup
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f'P1-{model_name} Epoch {epoch+1}/{config.p1_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(config.device)
            labels = labels.to(config.device)

            # CNN提取全局特征（用于分类）
            features = cnn_model(images, return_spatial=False)  # (N, 768)

            # 计算损失：ArcFace + Center Margin
            total_loss, loss_dict = criterion(features, labels)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'arc': f"{loss_dict['arcface']:.3f}",
                'ctr': f"{loss_dict['center']:.3f}",
                'acc': f"{loss_dict['acc']*100:.1f}%",
                'warmup': f"{loss_dict.get('warmup', 0):.2f}"
            })

            if writer is not None:
                global_step = epoch * len(loader) + batch_idx
                writer.add_scalar(f'Phase1_{model_name}/BatchLoss', total_loss.item(), global_step)
                writer.add_scalar(f'Phase1_{model_name}/ArcFaceLoss', loss_dict['arcface'], global_step)
                writer.add_scalar(f'Phase1_{model_name}/CenterLoss', loss_dict['center'], global_step)
                writer.add_scalar(f'Phase1_{model_name}/Accuracy', loss_dict['acc'], global_step)

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if writer is not None:
            writer.add_scalar(f'Phase1_{model_name}/EpochLoss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': cnn_model.state_dict(),
                'criterion': criterion.state_dict()
            }, os.path.join(config.save_dir, f'{model_name}_phase1_best.pth'))

        if early_stop(avg_loss, mode='min'):
            break

    return best_loss

def train_phase2(cnn_palm, cnn_vein, config, writer=None):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载第一阶段训练好的CNN模型权重
    for model, name in [(cnn_palm, 'cnn_palm'), (cnn_vein, 'cnn_vein')]:
        ckpt_path = os.path.join(config.save_dir, f'{name}_phase1_best.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model'])
            print(f" Loaded {name} checkpoint from {ckpt_path}")
        else:
            print(f" Warning: {name} checkpoint not found at {ckpt_path}")

    train_ds = PairDataset(config.palm_dir2, config.vein_dir2,
                          get_transforms(strong=True), split='train')
    val_ds = PairDataset(config.palm_dir2, config.vein_dir2,
                        get_transforms(strong=False), split='val')

    actual_num_classes = train_ds.num_classes

    if actual_num_classes != config.num_classes:
        num_classes_for_model = actual_num_classes
    else:
        num_classes_for_model = config.num_classes

    train_loader = DataLoader(train_ds, batch_size=config.p2_batch,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.p2_batch,
                           shuffle=False, num_workers=4, pin_memory=True)

    fusion_model = Stage2FusionCA(
        in_dim_global_palm=768,   # CNN全局特征维度
        in_dim_global_vein=768,   # CNN全局特征维度
        in_dim_local_palm=768,    # CNN空间特征维度
        in_dim_local_vein=768,    # CNN空间特征维度
        out_dim_global=256,
        out_dim_local=256,
        use_spatial_fusion=True,  # 使用空间注意力融合
        final_l2norm=True,
        with_arcface=True,
        num_classes=num_classes_for_model,
        arcface_s=30.0,
        arcface_m=0.20
    ).to(config.device)

    freeze_stage1 = False # 设置为True以冻结第一阶段的CNN编码器

    if freeze_stage1:
        for model in [cnn_palm, cnn_vein]:
            for param in model.parameters():
                param.requires_grad = False
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=config.p2_lr, weight_decay=1e-2)
    else:
        # 联合训练：融合模块用较大学习率，CNN编码器用较小学习率微调
        optimizer = torch.optim.Adam([
            {'params': fusion_model.parameters(), 'lr': config.p2_lr},
            {'params': cnn_palm.parameters(), 'lr': config.p2_enc_lr},
            {'params': cnn_vein.parameters(), 'lr': config.p2_enc_lr}
        ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.p2_epochs)

    criterion = get_stage2_loss(
        num_classes=num_classes_for_model,
        feat_dim=512,  # out_dim_global + out_dim_local = 256 + 256
        lambda_balance=0.1,  # 门控平衡损失权重
        lambda_diversity=0.05,  # 门控多样性损失权重
        mode='standard'  # 使用标准模式：分类 + 门控正则化
    ).to(config.device)

    early_stop = EarlyStopping(patience=config.p2_patience, min_delta=0.1)

    best_acc = 0.0
    for epoch in range(config.p2_epochs):
        # 设置模型训练/评估模式
        if freeze_stage1:
            for model in [cnn_palm, cnn_vein]:
                model.eval()
        else:
            for model in [cnn_palm, cnn_vein]:
                model.train()
        fusion_model.train()

        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f'P2 Epoch {epoch+1}/{config.p2_epochs}')

        for palm_img, vein_img, labels in pbar:
            palm_img = palm_img.to(config.device)
            vein_img = vein_img.to(config.device)
            labels = labels.to(config.device)

            if freeze_stage1:
                with torch.no_grad():
                    # CNN提取全局特征（return_spatial=False）
                    palm_global = cnn_palm(palm_img, return_spatial=False)
                    vein_global = cnn_vein(vein_img, return_spatial=False)
                    # CNN提取局部空间特征（return_spatial=True）
                    palm_local = cnn_palm(palm_img, return_spatial=True)
                    vein_local = cnn_vein(vein_img, return_spatial=True)
            else:
                # CNN提取全局特征（return_spatial=False）
                palm_global = cnn_palm(palm_img, return_spatial=False)
                vein_global = cnn_vein(vein_img, return_spatial=False)
                # CNN提取局部空间特征（return_spatial=True）
                palm_local = cnn_palm(palm_img, return_spatial=True)
                vein_local = cnn_vein(vein_img, return_spatial=True)

            optimizer.zero_grad()
            logits, fused_feat, details = fusion_model(
                palm_global, vein_global,
                palm_local, vein_local,
                labels
            )

            loss, loss_dict = criterion(logits, labels, fused_feat, details)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            _, pred = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            train_acc = 100. * train_correct / train_total

        scheduler.step()
        train_loss = train_loss_sum / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        for model in [cnn_palm, cnn_vein, fusion_model]:
            model.eval()

        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for palm_img, vein_img, labels in val_loader:
                palm_img = palm_img.to(config.device)
                vein_img = vein_img.to(config.device)
                labels = labels.to(config.device)

                # CNN提取全局和局部特征
                palm_global = cnn_palm(palm_img, return_spatial=False)
                vein_global = cnn_vein(vein_img, return_spatial=False)
                palm_local = cnn_palm(palm_img, return_spatial=True)
                vein_local = cnn_vein(vein_img, return_spatial=True)

                # 使用 ArcFace 获取 logits（传入 labels）
                logits, fused_feat, _ = fusion_model(
                    palm_global, vein_global,
                    palm_local, vein_local,
                    labels
                )
                loss, _ = criterion(logits, labels)

                _, pred = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
                val_loss_sum += loss.item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # 计算生物识别指标（每5个epoch计算一次）
        bio_metrics = None
        if epoch % 5 == 0 or epoch == config.p2_epochs - 1:
            # 提取特征
            all_features, all_labels = [], []
            sample_count, max_samples = 0, 300

            with torch.no_grad():
                for palm, vein, labels in val_loader:
                    if sample_count >= max_samples:
                        break
                    palm, vein = palm.to(config.device), vein.to(config.device)

                    # CNN提取全局和局部特征
                    palm_global = cnn_palm(palm, return_spatial=False)
                    vein_global = cnn_vein(vein, return_spatial=False)
                    palm_local = cnn_palm(palm, return_spatial=True)
                    vein_local = cnn_vein(vein, return_spatial=True)

                    # 融合特征
                    fused_feat, _ = fusion_model(palm_global, vein_global, palm_local, vein_local)

                    all_features.append(fused_feat.cpu().numpy())
                    all_labels.append(labels.numpy())
                    sample_count += len(labels)

            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # 计算genuine和impostor分数
            genuine_scores, impostor_scores = [], []
            id_to_indices = {}
            for idx, label in enumerate(all_labels):
                id_to_indices.setdefault(int(label), []).append(idx)

            # Genuine scores
            for indices in id_to_indices.values():
                if len(indices) >= 2:
                    for i, j in itertools.combinations(indices, 2):
                        genuine_scores.append(cosine_similarity([all_features[i]], [all_features[j]])[0][0])

            # Impostor scores
            all_identities = list(id_to_indices.keys())
            for i, label in enumerate(all_identities):
                for other_label in all_identities[i+1:i+6]:
                    for idx1 in id_to_indices[label][:2]:
                        for idx2 in id_to_indices[other_label][:2]:
                            impostor_scores.append(cosine_similarity([all_features[idx1]], [all_features[idx2]])[0][0])

            if len(genuine_scores) > 0 and len(impostor_scores) > 0:
                genuine_scores = np.array(genuine_scores)
                impostor_scores = np.array(impostor_scores)
                scores = np.concatenate([genuine_scores, impostor_scores])
                labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]).astype(int)

                try:
                    eer, _ = compute_eer(scores, labels, is_similarity=True, return_threshold=True)
                    _, _, _, auc_score = roc_auc(scores, labels, is_similarity=True)
                    tar_1 = tar_at_far(scores, labels, target_far=0.01, is_similarity=True)

                    bio_metrics = {
                        'EER': eer,
                        'AUC': auc_score,
                        'TAR@1%': tar_1['TAR'],
                        'genuine_mean': np.mean(genuine_scores),
                        'impostor_mean': np.mean(impostor_scores)
                    }
                except Exception as e:
                    print(f" Warning: Failed to compute metrics: {e}")

        print(f"\nEpoch {epoch+1}/{config.p2_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if bio_metrics is not None:
            print(f"  EER: {bio_metrics['EER']:.4f}, AUC: {bio_metrics['AUC']:.4f}")
            print(f"  TAR@1%FAR: {bio_metrics['TAR@1%']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # TensorBoard 记录
        if writer is not None:
            writer.add_scalar('Phase2/TrainLoss', train_loss, epoch)
            writer.add_scalar('Phase2/TrainAcc', train_acc, epoch)
            writer.add_scalar('Phase2/ValLoss', val_loss, epoch)
            writer.add_scalar('Phase2/ValAcc', val_acc, epoch)
            writer.add_scalar('Phase2/LearningRate', current_lr, epoch)

            # 记录生物识别指标
            if bio_metrics is not None:
                writer.add_scalar('Phase2/EER', bio_metrics['EER'], epoch)
                writer.add_scalar('Phase2/AUC', bio_metrics['AUC'], epoch)
                writer.add_scalar('Phase2/TAR@1%FAR', bio_metrics['TAR@1%'], epoch)
                writer.add_scalar('Phase2/GenuineMean', bio_metrics['genuine_mean'], epoch)
                writer.add_scalar('Phase2/ImpostorMean', bio_metrics['impostor_mean'], epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'cnn_palm': cnn_palm.state_dict(),
                'cnn_vein': cnn_vein.state_dict(),
                'fusion': fusion_model.state_dict(),
                'best_acc': best_acc
            }, os.path.join(config.save_dir, 'stage2_best.pth'))
            print(f" Saved best model (Val Acc: {best_acc:.2f}%)")

        if early_stop(val_acc, mode='max'):
            print("\n Early stopping triggered")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_acc

def main():
    writer = SummaryWriter(log_dir='outputs/runs/phase2_experiment')

    cnn_palm = ConvNeXt(
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ).to(config.device)

    cnn_vein = ConvNeXt(
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ).to(config.device)

    skip_stage1 = True  #设置为True以跳过第一阶段训练

    if not skip_stage1:
        cnn_palm_loss = train_phase1_cnn(cnn_palm, config, writer, 'cnn_palm')
        print(f" CNN Palm best loss: {cnn_palm_loss:.4f}")
        cnn_vein_loss = train_phase1_cnn(cnn_vein, config, writer, 'cnn_vein')
        print(f" CNN Vein best loss: {cnn_vein_loss:.4f}")

    best_acc = train_phase2(cnn_palm, cnn_vein, config, writer)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")

    writer.close()

if __name__ == '__main__':
    main()