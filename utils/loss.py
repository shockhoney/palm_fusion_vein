import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================================
# 第一阶段：对比学习损失（Triplet Loss）
# ====================================================================================
class TripletLoss(nn.Module):
    """
    三元组损失 - 用于 Stage1 预训练
    拉近 anchor 和 positive，推远 anchor 和 negative
    """
    def __init__(self, margin=0.5, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining  # 'none', 'hard', 'semi-hard'

    def forward(self, feat_anchor, feat_positive, feat_negative):
        # L2 归一化
        feat_anchor = F.normalize(feat_anchor, p=2, dim=1)
        feat_positive = F.normalize(feat_positive, p=2, dim=1)
        feat_negative = F.normalize(feat_negative, p=2, dim=1)

        # 计算余弦距离（1 - cosine_similarity）
        dist_ap = 1.0 - F.cosine_similarity(feat_anchor, feat_positive, dim=1)
        dist_an = 1.0 - F.cosine_similarity(feat_anchor, feat_negative, dim=1)

        # Triplet loss with margin
        triplet_loss = F.relu(dist_ap - dist_an + self.margin)

        # Hard mining（可选）
        if self.mining == 'hard':
            # 只保留困难样本（loss > 0）
            hard_triplets = triplet_loss > 0
            if hard_triplets.sum() > 0:
                triplet_loss = triplet_loss[hard_triplets].mean()
            else:
                triplet_loss = triplet_loss.mean()
        else:
            triplet_loss = triplet_loss.mean()

        return triplet_loss, dist_ap.mean(), dist_an.mean()


class IdentityLoss(nn.Module):
    """
    Identity Loss - 用于 Stage1
    通过简单的分类任务增强特征的判别性
    """
    def __init__(self, feat_dim, num_classes, s=16.0):
        super(IdentityLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        """
        Args:
            features: (N, feat_dim) 特征向量
            labels: (N,) 类别标签
        """
        # L2 归一化
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=0)

        # 计算余弦相似度并缩放
        logits = F.linear(features_norm, weight_norm) * self.s

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss, logits


class Stage1Loss(nn.Module):
    """
    Stage1 完整损失：Triplet Loss + Identity Loss
    根据论文图片设计
    """
    def __init__(self, feat_dim, num_classes,
                 triplet_margin=0.5,
                 triplet_mining='hard',
                 w_triplet=1.0,
                 w_identity=0.5,
                 identity_s=16.0):
        super(Stage1Loss, self).__init__()
        self.w_triplet = w_triplet
        self.w_identity = w_identity

        self.triplet_loss = TripletLoss(margin=triplet_margin, mining=triplet_mining)
        self.identity_loss = IdentityLoss(feat_dim, num_classes, s=identity_s)

    def forward(self, feat_anchor, feat_positive, feat_negative, labels):
        """
        Args:
            feat_anchor: (N, feat_dim) anchor特征
            feat_positive: (N, feat_dim) positive特征
            feat_negative: (N, feat_dim) negative特征
            labels: (N,) anchor的类别标签
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. Triplet Loss
        triplet_loss, dist_ap, dist_an = self.triplet_loss(feat_anchor, feat_positive, feat_negative)
        loss_dict['triplet'] = triplet_loss.item()
        loss_dict['dist_ap'] = dist_ap.item()
        loss_dict['dist_an'] = dist_an.item()

        # 2. Identity Loss（仅对 anchor 计算）
        identity_loss, logits = self.identity_loss(feat_anchor, labels)
        loss_dict['identity'] = identity_loss.item()

        # 计算分类准确率
        _, pred = torch.max(logits, 1)
        acc = (pred == labels).float().mean()
        loss_dict['identity_acc'] = acc.item()

        # 总损失
        total_loss = self.w_triplet * triplet_loss + self.w_identity * identity_loss
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ====================================================================================
# 第二阶段：多任务损失（适配 Stage2FusionCA）
# ====================================================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss（已集成在 Stage2FusionCA 中）
    这里提供独立版本以便灵活使用
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ModalityConsistencyLoss(nn.Module):
    """
    模态一致性损失 - 确保融合特征保留两个模态的信息
    通过对比全局特征和局部特征的融合结果
    """
    def __init__(self, temperature=0.07):
        super(ModalityConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, global_fused, local_fused):
        """
        Args:
            global_fused: 全局特征融合结果 (N, dim)
            local_fused: 局部特征融合结果 (N, dim)
        """
        # L2 归一化
        global_fused = F.normalize(global_fused, dim=1)
        local_fused = F.normalize(local_fused, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(global_fused, local_fused.t()) / self.temperature

        # 对角线应该是最大的（同一样本的全局和局部应该一致）
        labels = torch.arange(global_fused.size(0), device=global_fused.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class AttentionRegularizationLoss(nn.Module):
    """
    注意力正则化损失 - 鼓励注意力权重的多样性
    避免注意力权重退化（总是偏向某一个模态）
    """
    def __init__(self, epsilon=1e-8):
        super(AttentionRegularizationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, attention_weights_palm, attention_weights_vein):
        """
        Args:
            attention_weights_palm: 掌纹注意力权重 (N, C) 或 (N, 1, H, W)
            attention_weights_vein: 掌静脉注意力权重 (N, C) 或 (N, 1, H, W)
        """
        # 计算平均注意力权重
        mean_palm = attention_weights_palm.mean()
        mean_vein = attention_weights_vein.mean()

        # 鼓励权重接近 0.5（两个模态平衡）
        balance_loss = (mean_palm - 0.5) ** 2 + (mean_vein - 0.5) ** 2

        # 鼓励权重有一定的方差（避免所有位置权重相同）
        var_palm = attention_weights_palm.var()
        var_vein = attention_weights_vein.var()
        diversity_loss = torch.exp(-var_palm) + torch.exp(-var_vein)

        return balance_loss + 0.1 * diversity_loss


class CenterLoss(nn.Module):
    """
    Center Loss - 拉近同类样本的特征
    配合 ArcFace 使用可以进一步提升类内紧凑性
    """
    def __init__(self, num_classes, feat_dim, lambda_c=0.01):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # 可学习的类中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        """
        Args:
            features: 特征向量 (N, feat_dim)
            labels: 类别标签 (N,)
        """
        # L2 归一化
        features = F.normalize(features, dim=1)
        centers = F.normalize(self.centers, dim=1)

        # 计算每个样本到其类中心的距离
        batch_size = features.size(0)
        centers_batch = centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / batch_size

        return self.lambda_c * loss


class Stage2FusionLoss(nn.Module):
    """
    Stage2 完整损失函数 - 适配 Stage2FusionCA 架构

    包含以下损失项：
    1. 分类损失（CrossEntropy 或 ArcFace）
    2. 模态一致性损失（可选）
    3. 注意力正则化损失（可选）
    4. Center Loss（可选）

    推荐配置：
    - 仅分类: w_cls=1.0, 其余为 0
    - 标准配置: w_cls=1.0, w_consistency=0.1, w_attention=0.05
    - 完整配置: w_cls=1.0, w_consistency=0.1, w_attention=0.05, w_center=0.01
    """
    def __init__(self,
                 num_classes,
                 feat_dim=512,
                 w_cls=1.0,
                 w_consistency=0.0,
                 w_attention=0.0,
                 w_center=0.0,
                 label_smoothing=0.1,
                 use_arcface=True):
        super(Stage2FusionLoss, self).__init__()

        self.w_cls = w_cls
        self.w_consistency = w_consistency
        self.w_attention = w_attention
        self.w_center = w_center
        self.use_arcface = use_arcface

        # 1. 分类损失（如果 Stage2FusionCA 中已包含 ArcFace，此处使用 CE）
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 2. 模态一致性损失
        if w_consistency > 0:
            self.consistency_loss = ModalityConsistencyLoss(temperature=0.07)

        # 3. 注意力正则化损失
        if w_attention > 0:
            self.attention_loss = AttentionRegularizationLoss()

        # 4. Center Loss
        if w_center > 0:
            self.center_loss = CenterLoss(num_classes, feat_dim, lambda_c=1.0)

    def forward(self, logits, labels, fused_feat=None, details=None):
        """
        Args:
            logits: 分类 logits (N, num_classes)
            labels: 标签 (N,)
            fused_feat: 融合后的特征向量 (N, feat_dim)，用于 Center Loss
            details: Stage2FusionCA 返回的中间结果字典，包含：
                - global['fused']: 全局融合特征 (N, G)
                - local['fused']: 局部融合特征 (N, L)
                - global['w_palm'], global['w_vein']: 全局注意力权重
                - local['w_palm'], local['w_vein']: 局部注意力权重
        """
        loss_dict = {}

        # 1. 分类损失
        cls_loss = self.ce_loss(logits, labels)
        loss_dict['cls'] = cls_loss.item()
        total_loss = self.w_cls * cls_loss

        # 2. 模态一致性损失
        if self.w_consistency > 0 and details is not None:
            global_fused = details['global']['fused']
            local_fused = details['local']['fused']
            consistency_loss = self.consistency_loss(global_fused, local_fused)
            loss_dict['consistency'] = consistency_loss.item()
            total_loss += self.w_consistency * consistency_loss

        # 3. 注意力正则化损失
        if self.w_attention > 0 and details is not None:
            # 全局注意力正则化
            global_attn_loss = self.attention_loss(
                details['global']['w_palm'],
                details['global']['w_vein']
            )
            # 局部注意力正则化
            local_attn_loss = self.attention_loss(
                details['local']['w_palm'],
                details['local']['w_vein']
            )
            attention_loss = (global_attn_loss + local_attn_loss) / 2.0
            loss_dict['attention'] = attention_loss.item()
            total_loss += self.w_attention * attention_loss

        # 4. Center Loss
        if self.w_center > 0 and fused_feat is not None:
            center_loss = self.center_loss(fused_feat, labels)
            loss_dict['center'] = center_loss.item()
            total_loss += self.w_center * center_loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ====================================================================================
# 便捷函数：创建推荐的损失函数
# ====================================================================================

def get_stage1_loss(feat_dim, num_classes,
                    triplet_margin=0.5, triplet_mining='hard',
                    w_triplet=1.0, w_identity=0.5,
                    identity_s=16.0):
    """
    获取 Stage1 推荐损失函数（Triplet Loss + Identity Loss）

    Args:
        feat_dim: 特征维度（ViT: 192, CNN: 768）
        num_classes: 类别数（用于 Identity Loss）
        triplet_margin: Triplet margin（建议 0.3-1.0）
        triplet_mining: 困难样本挖掘策略 ('none', 'hard')
        w_triplet: Triplet Loss权重
        w_identity: Identity Loss权重
        identity_s: Identity Loss的缩放因子
    """
    return Stage1Loss(
        feat_dim=feat_dim,
        num_classes=num_classes,
        triplet_margin=triplet_margin,
        triplet_mining=triplet_mining,
        w_triplet=w_triplet,
        w_identity=w_identity,
        identity_s=identity_s
    )


def get_stage2_loss(num_classes, feat_dim=512, mode='standard'):
    """
    获取 Stage2 推荐损失函数

    Args:
        num_classes: 类别数
        feat_dim: 特征维度（out_dim_global + out_dim_local）
        mode: 损失模式
            - 'simple': 仅分类损失
            - 'standard': 分类 + 一致性
            - 'full': 分类 + 一致性 + 注意力正则化
            - 'advanced': 完整配置（包含 Center Loss）
    """
    if mode == 'simple':
        return Stage2FusionLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            w_cls=1.0,
            w_consistency=0.0,
            w_attention=0.0,
            w_center=0.0,
            label_smoothing=0.0  # 🔧 关闭 label smoothing
        )
    elif mode == 'standard':
        return Stage2FusionLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            w_cls=1.0,
            w_consistency=0.1,
            w_attention=0.0,
            w_center=0.0,
            label_smoothing=0.1
        )
    elif mode == 'full':
        return Stage2FusionLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            w_cls=1.0,
            w_consistency=0.1,
            w_attention=0.05,
            w_center=0.0,
            label_smoothing=0.1
        )
    elif mode == 'advanced':
        return Stage2FusionLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            w_cls=1.0,
            w_consistency=0.1,
            w_attention=0.05,
            w_center=0.01,
            label_smoothing=0.1
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from ['simple', 'standard', 'full', 'advanced']")
