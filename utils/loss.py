import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m_max=0.50):
        super().__init__()
        self.s = s
        self.m_max = m_max 
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels, m_eff=None):
        if m_eff is None:
            m_eff = self.m_max
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, w)  
        if m_eff == 0:
            return cosine * self.s    
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        cos_m = math.cos(m_eff)
        sin_m = math.sin(m_eff)
        phi = cosine * cos_m - sine * sin_m
        th = math.cos(math.pi - m_eff)
        mm = math.sin(math.pi - m_eff) * m_eff
        phi = torch.where(cosine > th, phi, cosine - mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1 - one_hot) * cosine)

        return output * self.s

class CenterMarginLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_margin=0.01, margin=2.0):
        super().__init__()
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers, dim=1)
        centers_batch = centers[labels]
        center_loss = ((x - centers_batch) ** 2).sum(dim=1).mean()
        dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(len(centers), device=x.device, dtype=bool)
        margin_loss = F.relu(self.margin - dist[mask]).pow(2).mean()

        return center_loss + self.lambda_margin * margin_loss


class Stage1Loss(nn.Module):
    def __init__(self, feat_dim, num_classes, s=30.0, m_max=0.30,
                 lambda_center=0.005, lambda_margin=0.001, margin=2.0,
                 warmup_epochs=20, center_start_epoch=5):
        super().__init__()
        self.s = s
        self.m_max = m_max
        self.lambda_center = lambda_center
        self.warmup_epochs = warmup_epochs
        self.center_start_epoch = center_start_epoch
        self.arcface = ArcFaceLoss(feat_dim, num_classes, s=s, m_max=m_max)
        self.center_margin = CenterMarginLoss(num_classes, feat_dim, lambda_margin, margin)
        self.ce = nn.CrossEntropyLoss()

        self.current_epoch = 0  

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, features, labels):
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
        m_eff = self.m_max * warmup_factor
        logits = self.arcface(features, labels, m_eff=m_eff)
        cls_loss = self.ce(logits, labels)
        if self.current_epoch >= self.center_start_epoch:
            cm_loss = self.center_margin(features, labels)
            total = cls_loss + self.lambda_center * cm_loss
        else:
            cm_loss = torch.tensor(0.0, device=features.device)
            total = cls_loss
        acc = (logits.argmax(1) == labels).float().mean()
        return total, {
            'total': total.item(),
            'arcface': cls_loss.item(),
            'center': cm_loss.item() if isinstance(cm_loss, torch.Tensor) else 0.0,
            'acc': acc.item(),
            'm_eff': m_eff, 
            'warmup': warmup_factor
        }

class GateRegularization(nn.Module):
    def __init__(self, lambda_balance=0.1, lambda_diversity=0.05):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity

    def forward(self, w_a, w_b):
        balance = ((w_a - 0.5) ** 2 + (w_b - 0.5) ** 2).mean()
        std_a = w_a.std(dim=1).mean()
        std_b = w_b.std(dim=1).mean()
        diversity = torch.exp(-std_a) + torch.exp(-std_b)

        return self.lambda_balance * balance + self.lambda_diversity * diversity


class Stage2Loss(nn.Module):
    def __init__(self, num_classes, feat_dim=512,
                 lambda_balance=0.1, lambda_diversity=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gate_reg = GateRegularization(lambda_balance, lambda_diversity) \
                        if lambda_balance > 0 or lambda_diversity > 0 else None

    def forward(self, logits, labels, fused_feat=None, details=None):
        cls_loss = self.ce(logits, labels)
        total = cls_loss

        loss_dict = {
            'cls': cls_loss.item(),
            'acc': (logits.argmax(1) == labels).float().mean().item()
        }

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


def get_stage1_loss(feat_dim, num_classes, s=30.0, m_max=0.30,
                    lambda_center=0.005, lambda_margin=0.001, margin=2.0,
                    warmup_epochs=20, center_start_epoch=5):
    return Stage1Loss(feat_dim, num_classes, s, m_max,
                     lambda_center, lambda_margin, margin,
                     warmup_epochs, center_start_epoch)

def get_stage2_loss(num_classes, feat_dim=512,
                    lambda_balance=0.1, lambda_diversity=0.05, mode='standard'):
    if mode == 'simple':
        return Stage2Loss(num_classes, feat_dim, 0.0, 0.0)
    else:
        return Stage2Loss(num_classes, feat_dim, lambda_balance, lambda_diversity)
