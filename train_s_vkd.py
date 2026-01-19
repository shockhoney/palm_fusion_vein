import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.datasets_txt import PairTxtDataset
from utils.metrics import compute_eer, tar_at_far
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Student model VKD training")
    parser.add_argument('--train_list', type=str, default='polyu_phase2_train.txt', help='训练列表文件路径')
    parser.add_argument('--val_list', type=str, default='polyu_phase2_val.txt', help='验证列表文件路径')
    parser.add_argument('--teacher_model', type=str, default='teacher_best.pth', help='教师模型权重路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--temperature', type=float, default=4.0, help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.5, help='蒸馏损失占比系数')
    parser.add_argument('--save_dir', type=str, default='output', help='模型保存目录')
    args = parser.parse_args()

    # 设备配置：优先使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 根据训练列表文件确定类别数（身份数量）
    label_set = set()
    with open(args.train_list, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                label = int(parts[2])
                label_set.add(label)
    num_classes = len(label_set)
    print(f"训练集包含 {num_classes} 个类别（身份）。")

    # 初始化教师模型和学生模型（假设模型类已定义于 models.py 等模块）
    from models import TeacherNet, StudentNet
    teacher_model = TeacherNet(num_classes=num_classes)
    student_model = StudentNet(num_classes=num_classes)
    # 加载教师模型权重并将教师模型置为评估模式，冻结其参数
    teacher_model.load_state_dict(torch.load(args.teacher_model, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model.to(device)

    # 定义损失函数：分类交叉熵损失 + 蒸馏(KL散度)损失
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    # 优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)

    # 准备数据集和数据加载器
    from torchvision import transforms
    transform_palm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # 可根据需要添加 transforms.Normalize 进行归一化
    ])
    transform_vein = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # 同样可添加 Normalize
    ])
    train_dataset = PairTxtDataset(args.train_list, transform_palm=transform_palm, transform_vein=transform_vein)
    val_dataset = PairTxtDataset(args.val_list, transform_palm=transform_palm, transform_vein=transform_vein)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 开始训练循环
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        running_loss = 0.0
        for palm_imgs, vein_imgs, labels in train_loader:
            palm_imgs = palm_imgs.to(device)
            vein_imgs = vein_imgs.to(device)
            labels = labels.to(device)
            # 教师模型前向传播（提供软目标），不计算梯度
            with torch.no_grad():
                teacher_logits = teacher_model(palm_imgs, vein_imgs)
            # 学生模型前向传播
            student_logits = student_model(palm_imgs, vein_imgs)
            # 计算损失：分类损失 + 蒸馏损失
            loss_cls = criterion_cls(student_logits, labels)
            T = args.temperature
            student_log_probs = F.log_softmax(student_logits / T, dim=1)
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            loss_kd = criterion_kd(student_log_probs, teacher_probs) * (T * T)
            # 按权重合并两种损失
            loss = args.alpha * loss_kd + (1 - args.alpha) * loss_cls

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * palm_imgs.size(0)
        # 计算当轮平均损失
        epoch_loss = running_loss / len(train_dataset)

        # 每5个epoch进行一次验证评估
        if epoch % 5 == 0:
            student_model.eval()
            all_feats = []
            all_labels = []
            # 遍历验证集，提取每个样本的融合特征
            with torch.no_grad():
                for palm_imgs, vein_imgs, labels in val_loader:
                    palm_imgs = palm_imgs.to(device)
                    vein_imgs = vein_imgs.to(device)
                    # 提取学生模型的融合特征向量
                    if hasattr(student_model, "get_features"):
                        fused_feat = student_model.get_features(palm_imgs, vein_imgs)
                    elif hasattr(student_model, "feature_extractor"):
                        fused_feat = student_model.feature_extractor(palm_imgs, vein_imgs)
                    elif hasattr(student_model, "backbone_palm") and hasattr(student_model, "backbone_vein"):
                        # 如果模型有单独的Palm和Vein子网络
                        palm_feat = student_model.backbone_palm(palm_imgs)
                        vein_feat = student_model.backbone_vein(vein_imgs)
                        # 将Palm和Vein特征融合（若有专门融合层则使用，没有则直接拼接）
                        if hasattr(student_model, "fuse_layer"):
                            fused_feat = student_model.fuse_layer(torch.cat([palm_feat, vein_feat], dim=1))
                        else:
                            fused_feat = torch.cat([palm_feat, vein_feat], dim=1)
                    else:
                        # 如果没有特定方法，假定 student_model(palm, vein) 直接返回融合特征
                        fused_feat = student_model(palm_imgs, vein_imgs)
                    # 对特征向量进行L2归一化
                    fused_feat = F.normalize(fused_feat, dim=1)
                    all_feats.append(fused_feat.cpu().numpy())
                    all_labels.extend(labels.numpy().tolist())
            # 构建相似度矩阵并计算验证指标
            all_feats = np.vstack(all_feats)  # shape: (N, feature_dim)
            all_labels = np.array(all_labels)
            sim_matrix = np.dot(all_feats, all_feats.T)  # 计算余弦相似度矩阵（特征已归一化）
            N = sim_matrix.shape[0]
            genuine_scores = []
            impostor_scores = []
            for i in range(N):
                for j in range(i + 1, N):
                    if all_labels[i] == all_labels[j]:
                        genuine_scores.append(sim_matrix[i, j])
                    else:
                        impostor_scores.append(sim_matrix[i, j])
            genuine_scores = np.array(genuine_scores)
            impostor_scores = np.array(impostor_scores)
            # 计算 EER 和各指定 FAR 下的 TAR
            val_eer = compute_eer(genuine_scores, impostor_scores)
            far_points = [1e-1,1e-2,1e-3, 1e-4,1e-5]  # 可根据需求添加 FAR=1e-2 等
            tar_list = []
            for far in far_points:
                tar_val = tar_at_far(genuine_scores, impostor_scores, far)
                tar_list.append(f"TAR@FAR={far:.0e}: {tar_val * 100:.2f}%")
            # 打印验证集评估结果
            print(f"Epoch [{epoch}/{args.epochs}] Loss: {epoch_loss:.4f}, Val EER: {val_eer * 100:.2f}%, " + ", ".join(tar_list))
            student_model.train()
        else:
            # 非评估epoch仅输出训练损失
            print(f"Epoch [{epoch}/{args.epochs}] Loss: {epoch_loss:.4f}")
    # 保存最终学生模型权重
    final_path = os.path.join(args.save_dir, "student_final.pth")
    torch.save(student_model.state_dict(), final_path)
    print(f"训练结束，学生模型已保存: {final_path}")

if __name__ == "__main__":
    main()
