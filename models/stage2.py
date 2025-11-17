import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        B, D = feat_a.shape
        q_a = self.q_proj(feat_a).view(B, self.num_heads, self.head_dim)
        k_b = self.k_proj(feat_b).view(B, self.num_heads, self.head_dim)
        v_b = self.v_proj(feat_b).view(B, self.num_heads, self.head_dim)

        attn_a = (q_a * k_b).sum(dim=-1, keepdim=True) * self.scale 
        attn_a = F.softmax(attn_a, dim=1)
        out_a = (attn_a * v_b).view(B, -1) 
        enhanced_a = self.norm(feat_a + self.out_proj(out_a))

        q_b = self.q_proj(feat_b).view(B, self.num_heads, self.head_dim)
        k_a = self.k_proj(feat_a).view(B, self.num_heads, self.head_dim)
        v_a = self.v_proj(feat_a).view(B, self.num_heads, self.head_dim)

        attn_b = (q_b * k_a).sum(dim=-1, keepdim=True) * self.scale 
        attn_b = F.softmax(attn_b, dim=1)
        out_b = (attn_b * v_a).view(B, -1) 
        enhanced_b = self.norm(feat_b + self.out_proj(out_b))

        return enhanced_a, enhanced_b


class ChannelAttentionFusion(nn.Module):
    """通道注意力融合：为两个特征生成通道权重并融合"""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(dim // reduction, 16)

        self.attention = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * dim)
        )

        # 初始化
        for m in self.attention:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """通道注意力融合

        Args:
            feat_a: (N, C) 特征 A
            feat_b: (N, C) 特征 B

        Returns:
            fused: (N, C) 融合后的特征
            w_a: (N, C) 特征 A 的权重
            w_b: (N, C) 特征 B 的权重
        """
        concat = torch.cat([feat_a, feat_b], dim=1)  # (N, 2C)
        attn_logits = self.attention(concat)  # (N, 2C)
        attn_logits = attn_logits.view(feat_a.size(0), 2, -1)  # (N, 2, C)
        attn_weights = F.softmax(attn_logits, dim=1)  # (N, 2, C)
        w_a, w_b = attn_weights[:, 0], attn_weights[:, 1]  # (N, C)
        fused = w_a * feat_a + w_b * feat_b  # (N, C)
        return fused, w_a, w_b


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(features, weight)

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class Stage2Fusion(nn.Module):
    def __init__(self,
                 in_dim_global_palm: int = 768,
                 in_dim_global_vein: int = 768,
                 in_dim_local_palm: int = 768,
                 in_dim_local_vein: int = 768,
                 out_dim_local: int = 256,
                 final_l2norm: bool = True,
                 with_arcface: bool = False,
                 num_classes: int = 0,
                 arcface_s: float = 30.0,
                 arcface_m: float = 0.30):
        super().__init__()
        self.final_l2norm = final_l2norm
        self.with_arcface = with_arcface

        assert in_dim_global_palm == in_dim_global_vein, "Global dims must match"
        global_dim = in_dim_global_palm

        self.global_cross_attn = CrossModalAttention(global_dim, num_heads=8)
        self.global_channel_fusion = ChannelAttentionFusion(global_dim, reduction=4)

        if in_dim_local_palm != out_dim_local:
            self.local_align_palm = nn.Linear(in_dim_local_palm, out_dim_local)
        else:
            self.local_align_palm = nn.Identity()

        if in_dim_local_vein != out_dim_local:
            self.local_align_vein = nn.Linear(in_dim_local_vein, out_dim_local)
        else:
            self.local_align_vein = nn.Identity()

        self.local_cross_attn = CrossModalAttention(out_dim_local, num_heads=8)
        self.local_channel_fusion = ChannelAttentionFusion(out_dim_local, reduction=4)

        self.final_dim = global_dim + out_dim_local

        if with_arcface:
            assert num_classes > 0, "num_classes must be positive when with_arcface=True"
            self.arcface = ArcFaceHead(self.final_dim, num_classes, arcface_s, arcface_m)
        else:
            self.arcface = None

    def forward(self,
                palm_global: torch.Tensor,
                vein_global: torch.Tensor,
                palm_local: torch.Tensor,
                vein_local: torch.Tensor,
                labels: torch.Tensor = None):

        # 全局特征融合
        global_palm_enhanced, global_vein_enhanced = self.global_cross_attn(palm_global, vein_global)
        global_fused, global_w_palm, global_w_vein = self.global_channel_fusion(
            global_palm_enhanced, global_vein_enhanced
        )

        # 处理局部特征维度
        if palm_local.dim() == 4:  # (N, C, H, W)
            palm_local = palm_local.mean(dim=[2, 3])  # (N, C)
        if vein_local.dim() == 4:  # (N, C, H, W)
            vein_local = vein_local.mean(dim=[2, 3])  # (N, C)

        palm_local_aligned = self.local_align_palm(palm_local)  # (N, out_dim_local)
        vein_local_aligned = self.local_align_vein(vein_local)  # (N, out_dim_local)

        # 局部特征融合
        local_palm_enhanced, local_vein_enhanced = self.local_cross_attn(
            palm_local_aligned, vein_local_aligned
        )

        local_fused, local_w_palm, local_w_vein = self.local_channel_fusion(
            local_palm_enhanced, local_vein_enhanced
        )

        # 最终融合
        fused_feat = torch.cat([global_fused, local_fused], dim=1)

        if self.final_l2norm:
            fused_feat = F.normalize(fused_feat, dim=1)

        # 构造 details，包含 gate 权重
        details = {
            'global': {
                'w_palm': global_w_palm,
                'w_vein': global_w_vein
            },
            'local': {
                'w_palm': local_w_palm,
                'w_vein': local_w_vein
            },
            'global_fused': global_fused,
            'local_fused': local_fused,
            'final_fused': fused_feat
        }

        if self.arcface is not None and labels is not None:
            logits = self.arcface(fused_feat, labels)
            return logits, fused_feat, details
        else:
            return fused_feat, details
