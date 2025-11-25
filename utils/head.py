import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):

    def __init__(self, feature_dim, num_classes, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):

        x = F.normalize(features, dim=1)            
        W = F.normalize(self.weight, dim=1)          

        cos_theta = F.linear(x, W)                  
        cos_theta = cos_theta.clamp(-1.0, 1.0)     
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cond = (cos_theta > 0).float()
            cos_theta_m = cond * cos_theta_m + (1.0 - cond) * cos_theta
        else:
            cond = (cos_theta > self.th).float()
            cos_theta_m = cond * cos_theta_m + (1.0 - cond) * (cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)      
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        logits = logits * self.s                    

        return logits
