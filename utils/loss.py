import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Stage-1 Loss: ArcFace + Center Margin
# ============================================================

class ArcFaceLoss(nn.Module):
    """ArcFace: Additive Angular Margin Loss"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # Normalize features and weights
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)

        # Compute cosine similarity
        cosine = F.linear(x, w)
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1 - one_hot) * cosine)

        return output * self.s


class CenterMarginLoss(nn.Module):
    """Center Loss + Inter-class Margin Constraint"""
    def __init__(self, num_classes, feat_dim, lambda_margin=0.01, margin=2.0):
        super().__init__()
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers, dim=1)

        # Intra-class compactness
        centers_batch = centers[labels]
        center_loss = ((x - centers_batch) ** 2).sum(dim=1).mean()

        # Inter-class separation
        dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(len(centers), device=x.device, dtype=bool)
        margin_loss = F.relu(self.margin - dist[mask]).pow(2).mean()

        return center_loss + self.lambda_margin * margin_loss


class Stage1Loss(nn.Module):
    """Stage-1: ArcFace + Center Margin Loss with Warmup"""
    def __init__(self, feat_dim, num_classes, s=30.0, m=0.50,
                 lambda_center=0.01, lambda_margin=0.001, margin=2.0):
        super().__init__()
        self.lambda_center = lambda_center
        self.arcface = ArcFaceLoss(feat_dim, num_classes, s, m)
        self.center_margin = CenterMarginLoss(num_classes, feat_dim, lambda_margin, margin)
        self.ce = nn.CrossEntropyLoss()

        # 用于预热的标准分类层
        self.fc = nn.Linear(feat_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.current_epoch = 0  # 追踪当前epoch
        self.warmup_epochs = 10  # 前10个epoch用Softmax预热

    def set_epoch(self, epoch):
        """设置当前epoch用于warmup调度"""
        self.current_epoch = epoch

    def forward(self, features, labels):
        # 计算warmup系数 (前10个epoch: 0->1)
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)

        # Softmax logits (简单分类)
        softmax_logits = self.fc(F.normalize(features, dim=1))

        # ArcFace logits (角度间隔分类)
        arcface_logits = self.arcface(features, labels)

        # 混合logits: 前期用Softmax，后期用ArcFace
        logits = (1 - warmup_factor) * softmax_logits + warmup_factor * arcface_logits

        # 分类损失
        cls_loss = self.ce(logits, labels)

        # Center margin约束（从第5个epoch开始生效）
        if self.current_epoch >= 5:
            cm_loss = self.center_margin(features, labels)
            total = cls_loss + self.lambda_center * cm_loss
        else:
            cm_loss = torch.tensor(0.0, device=features.device)
            total = cls_loss

        # Metrics
        acc = (logits.argmax(1) == labels).float().mean()
        return total, {
            'total': total.item(),
            'arcface': cls_loss.item(),
            'center': cm_loss.item() if isinstance(cm_loss, torch.Tensor) else 0.0,
            'acc': acc.item(),
            'warmup': warmup_factor
        }


# ============================================================
# Stage-2 Loss: Classification + Gate Regularization
# ============================================================

class GateRegularization(nn.Module):
    """Regularize gate weights for balanced multi-modal fusion"""
    def __init__(self, lambda_balance=0.1, lambda_diversity=0.05):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity

    def forward(self, w_a, w_b):
        # Balance: encourage 0.5/0.5 split
        balance = ((w_a - 0.5) ** 2 + (w_b - 0.5) ** 2).mean()

        # Diversity: encourage channel-wise variation
        std_a = w_a.std(dim=1).mean()
        std_b = w_b.std(dim=1).mean()
        diversity = torch.exp(-std_a) + torch.exp(-std_b)

        return self.lambda_balance * balance + self.lambda_diversity * diversity


class Stage2Loss(nn.Module):
    """Stage-2: Classification + Gate Regularization"""
    def __init__(self, num_classes, feat_dim=512,
                 lambda_balance=0.1, lambda_diversity=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gate_reg = GateRegularization(lambda_balance, lambda_diversity) \
                        if lambda_balance > 0 or lambda_diversity > 0 else None

    def forward(self, logits, labels, fused_feat=None, details=None):
        # Classification loss
        cls_loss = self.ce(logits, labels)
        total = cls_loss

        loss_dict = {
            'cls': cls_loss.item(),
            'acc': (logits.argmax(1) == labels).float().mean().item()
        }

        # Gate regularization (if enabled and details provided)
        if self.gate_reg is not None and details is not None:
            g_reg = self.gate_reg(
                details['global']['w_palm'],
                details['global']['w_vein']
            )
            l_reg = self.gate_reg(
                details['local']['w_palm'],
                details['local']['w_vein']
            )
            gate_loss = (g_reg + l_reg) / 2
            total = total + gate_loss
            loss_dict['gate_reg'] = gate_loss.item()

        loss_dict['total'] = total.item()
        return total, loss_dict


# ============================================================
# Convenience Functions
# ============================================================

def get_stage1_loss(feat_dim, num_classes, s=30.0, m=0.50,
                    lambda_center=0.01, lambda_margin=0.001, margin=2.0):
    """Create Stage-1 loss: ArcFace + Center Margin

    Args:
        feat_dim: Feature dimension (768 for CNN)
        num_classes: Number of identity classes
        s: ArcFace scale factor (default: 30.0)
        m: ArcFace angular margin (default: 0.50)
        lambda_center: Weight for center margin loss (default: 0.01)
        lambda_margin: Weight for inter-class margin (default: 0.001)
        margin: Minimum inter-class distance (default: 2.0)
    """
    return Stage1Loss(feat_dim, num_classes, s, m,
                     lambda_center, lambda_margin, margin)


def get_stage2_loss(num_classes, feat_dim=512,
                    lambda_balance=0.1, lambda_diversity=0.05, mode='standard'):
    """Create Stage-2 loss: Classification + Gate Regularization

    Args:
        num_classes: Number of identity classes
        feat_dim: Fused feature dimension (default: 512)
        lambda_balance: Weight for gate balance loss (default: 0.1)
        lambda_diversity: Weight for gate diversity loss (default: 0.05)
        mode: 'simple' (no regularization) or 'standard' (with regularization)
    """
    if mode == 'simple':
        return Stage2Loss(num_classes, feat_dim, 0.0, 0.0)
    else:
        return Stage2Loss(num_classes, feat_dim, lambda_balance, lambda_diversity)
