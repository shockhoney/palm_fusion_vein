import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        B, D = feat_a.shape
        # A as Query, B as Key/Value
        q_a = self.q_proj(feat_a).view(B, self.num_heads, self.head_dim)
        k_b = self.k_proj(feat_b).view(B, self.num_heads, self.head_dim)
        v_b = self.v_proj(feat_b).view(B, self.num_heads, self.head_dim)

        attn_a = (q_a * k_b).sum(dim=-1, keepdim=True) * self.scale 
        attn_a = F.softmax(attn_a, dim=1)
        out_a = (attn_a * v_b).view(B, -1) 
        enhanced_a = self.norm(feat_a + self.out_proj(out_a))

        # B as Query, A as Key/Value
        q_b = self.q_proj(feat_b).view(B, self.num_heads, self.head_dim)
        k_a = self.k_proj(feat_a).view(B, self.num_heads, self.head_dim)
        v_a = self.v_proj(feat_a).view(B, self.num_heads, self.head_dim)

        attn_b = (q_b * k_a).sum(dim=-1, keepdim=True) * self.scale 
        attn_b = F.softmax(attn_b, dim=1)
        out_b = (attn_b * v_a).view(B, -1) 
        enhanced_b = self.norm(feat_b + self.out_proj(out_b))

        return enhanced_a, enhanced_b

class ChannelAttentionFusion(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(dim // reduction, 16)

        self.attention = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * dim)
        )
        # Init weights
        for m in self.attention:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        concat = torch.cat([feat_a, feat_b], dim=1)  # (N, 2C)
        attn_logits = self.attention(concat)         # (N, 2C)
        attn_logits = attn_logits.view(feat_a.size(0), 2, -1)  # (N, 2, C)
        attn_weights = F.softmax(attn_logits, dim=1)           # (N, 2, C)
        
        w_a, w_b = attn_weights[:, 0], attn_weights[:, 1]      # (N, C)
        fused = w_a * feat_a + w_b * feat_b                    # (N, C)
        return fused

class Stage2Fusion(nn.Module):
    def __init__(self,
                 in_dim_global: int = 256, 
                 out_dim_final: int = 512, 
                 final_l2norm: bool = True,
                ):
        super().__init__()
        self.final_l2norm = final_l2norm

        self.global_cross_attn = CrossModalAttention(in_dim_global, num_heads=8)
        self.global_channel_fusion = ChannelAttentionFusion(in_dim_global, reduction=4)

        # 将融合后的256维度映射到512维
        self.proj = nn.Linear(in_dim_global, out_dim_final)
        
        # 初始化投影层
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.out_dim = out_dim_final

    def forward(self, palm_global: torch.Tensor, vein_global: torch.Tensor):

        palm_enhanced, vein_enhanced = self.global_cross_attn(palm_global, vein_global)
        fused_feat = self.global_channel_fusion(palm_enhanced, vein_enhanced)
        fused_feat = self.proj(fused_feat)  

        if self.final_l2norm:
            fused_feat = F.normalize(fused_feat, dim=1)

        return fused_feat