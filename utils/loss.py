import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor, m_eff: float = None) -> torch.Tensor:
        if m_eff is None:
            m_eff = self.m
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, w)

        if m_eff == 0:
            return cosine * self.s

        cos_m = math.cos(m_eff)
        sin_m = math.sin(m_eff)
        th = math.cos(math.pi - m_eff)
        mm = math.sin(math.pi - m_eff) * m_eff

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class ArcFaceCELoss(nn.Module):

    def __init__(self,
                 feat_dim: int,
                 num_classes: int,
                 s: float = 30.0,
                 m: float = 0.30,
                 warmup_epochs: int = 0):
        super().__init__()
        self.m = m
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.head = ArcFaceHead(feat_dim, num_classes, s=s, m=m)

        self.ce = nn.CrossEntropyLoss()

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_margin(self) -> float:
        if self.warmup_epochs > 0:
            factor = min(1.0, float(self.current_epoch) / float(self.warmup_epochs))
        else:
            factor = 1.0
        return self.m * factor

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        m_eff = self._get_margin()
        logits = self.head(features, labels, m_eff=m_eff)
        cls_loss = self.ce(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        loss_dict = {
            'acc': acc,
            'logits': logits
        }
        return cls_loss, loss_dict
