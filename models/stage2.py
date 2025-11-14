import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAlign(nn.Module):
    """将输入向量线性投影到统一维度"""
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        if isinstance(self.bn, nn.BatchNorm1d):
            x = self.bn(x)
        return x


class ConvAlign2d(nn.Module):
    """1x1卷积+BN：用于空间特征图对齐"""
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        self.proj = nn.Sequential(*layers)
        for m in self.proj:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制：
    - palm 生成 Query，vein 生成 Key & Value
    - vein 生成 Query，palm 生成 Key & Value
    - 允许两个模态相互查询和交互

    输入：两路同维度向量 (N, dim)
    输出：交互后的两路向量 (N, dim)×2
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim 必须能被 num_heads 整除"

        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 前馈网络（可选，增强表达能力）
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, palm: torch.Tensor, vein: torch.Tensor):
        """
        Args:
            palm: (N, dim) 掌纹特征
            vein: (N, dim) 掌静脉特征
        Returns:
            palm_enhanced: (N, dim) 交互后的掌纹特征
            vein_enhanced: (N, dim) 交互后的掌静脉特征
        """
        B, D = palm.shape

        # ===== Palm 查询 Vein =====
        # Palm 作为 Q，Vein 作为 K, V
        q_palm = self.q_proj(palm).view(B, self.num_heads, self.head_dim)  # (B, H, D/H)
        k_vein = self.k_proj(vein).view(B, self.num_heads, self.head_dim)  # (B, H, D/H)
        v_vein = self.v_proj(vein).view(B, self.num_heads, self.head_dim)  # (B, H, D/H)

        # Attention: palm 从 vein 中提取信息
        attn_palm = (q_palm * k_vein).sum(dim=-1, keepdim=True) * self.scale  # (B, H, 1)
        attn_palm = F.softmax(attn_palm, dim=1)  # 归一化
        palm_from_vein = (attn_palm * v_vein).view(B, -1)  # (B, D)

        # 残差连接 + Layer Norm
        palm_enhanced = self.norm1(palm + self.dropout(self.out_proj(palm_from_vein)))

        # ===== Vein 查询 Palm =====
        # Vein 作为 Q，Palm 作为 K, V
        q_vein = self.q_proj(vein).view(B, self.num_heads, self.head_dim)
        k_palm = self.k_proj(palm).view(B, self.num_heads, self.head_dim)
        v_palm = self.v_proj(palm).view(B, self.num_heads, self.head_dim)

        # Attention: vein 从 palm 中提取信息
        attn_vein = (q_vein * k_palm).sum(dim=-1, keepdim=True) * self.scale  # (B, H, 1)
        attn_vein = F.softmax(attn_vein, dim=1)
        vein_from_palm = (attn_vein * v_palm).view(B, -1)  # (B, D)

        # 残差连接 + Layer Norm
        vein_enhanced = self.norm1(vein + self.dropout(self.out_proj(vein_from_palm)))

        # 前馈网络增强
        palm_enhanced = self.norm2(palm_enhanced + self.ffn(palm_enhanced))
        vein_enhanced = self.norm2(vein_enhanced + self.ffn(vein_enhanced))

        return palm_enhanced, vein_enhanced


class PairwiseChannelAttentionFusion(nn.Module):
    """
    输入：两路同维度向量 a, b （形状 (N, C)）
    思路：
      1) 拼接 [a, b] -> (N, 2C)
      2) MLP 产生每个通道在 a/b 两路上的权重 logits -> (N, 2C)
      3) 通道维度 reshape -> (N, 2, C)，在"路维"上做 softmax 得到 w_a, w_b
      4) 融合：fused = w_a * a + w_b * b
    优点：逐通道、权重和为 1，稳定可解释。
    """
    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = max(dim // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hid, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hid, 2 * dim, bias=True)
        )
        # 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape, "PairwiseChannelAttentionFusion: a/b 维度必须一致"
        x = torch.cat([a, b], dim=1)                # (N, 2C)
        logits = self.mlp(x)                        # (N, 2C)
        logits = logits.view(x.size(0), 2, -1)      # (N, 2, C)
        weights = F.softmax(logits, dim=1)          # (N, 2, C) -> 两路权重逐通道相加为 1
        wa, wb = weights[:, 0, :], weights[:, 1, :] # (N, C), (N, C)
        fused = wa * a + wb * b                     # (N, C)
        return fused, (wa, wb)


class ArcMarginProduct(nn.Module):
    """
    ArcFace / CosFace 类似的角度间隔分类头（默认 ArcFace）：
    输入向量与分类权重都做 L2 归一化，基于 cos(theta + m) 计算。
    """
    def __init__(self, in_features: int, out_features: int,
                 s: float = 64.0, m: float = 0.50,
                 easy_margin: bool = False):
        super().__init__()
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

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        # 归一化特征与权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (N, out)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # 输出：真实类用 phi，其余用原 cos
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class Stage2FusionCA(nn.Module):
    """
    第二阶段融合模型（改进版）：
    - 支持向量特征融合（通道注意力）
    - 支持空间特征融合（空间注意力）

    参数：
      - in_dim_global_palm / vein：第一阶段全局向量维度（默认 ViT embed_dim[-1]）
      - in_dim_local_palm / vein：第一阶段局部向量维度（默认 ConvNeXt dims[-1]）
      - out_dim_global / out_dim_local：对齐后的通道数（建议相同，如 256）
      - use_spatial_fusion：是否对局部特征使用空间注意力融合（保留空间信息）
      - final_l2norm：融合后是否 L2 归一化（建议 True，以便 ArcFace）
      - with_arcface：是否启用 ArcFace 分类头（可选）

    前向：
      forward_features(...) -> fused_feat（拼接后的融合向量）
      forward(..., labels)  -> ArcFace logits（若 with_arcface=True）
    """
    def __init__(self,
                 in_dim_global_palm: int = 192,
                 in_dim_global_vein: int = 192,
                 in_dim_local_palm: int = 768,
                 in_dim_local_vein: int = 768,
                 out_dim_global: int = 256,
                 out_dim_local: int = 256,
                 use_spatial_fusion: bool = False,
                 final_l2norm: bool = True,
                 with_arcface: bool = False,
                 num_classes: int = 0,
                 arcface_s: float = 64.0,
                 arcface_m: float = 0.50):
        super().__init__()

        self.use_spatial_fusion = use_spatial_fusion

        # 1) 全局特征对齐（向量 -> 向量）
        self.g_align_palm = FeatureAlign(in_dim_global_palm, out_dim_global, use_bn=False)
        self.g_align_vein = FeatureAlign(in_dim_global_vein, out_dim_global, use_bn=False)

        # 2) 局部特征对齐：根据是否使用空间融合选择不同的对齐方式
        if use_spatial_fusion:
            # 空间特征融合：用卷积对齐通道数
            self.l_align_palm = ConvAlign2d(in_dim_local_palm, out_dim_local, use_bn=True)
            self.l_align_vein = ConvAlign2d(in_dim_local_vein, out_dim_local, use_bn=True)
        else:
            # 向量特征融合：用全连接对齐维度
            self.l_align_palm = FeatureAlign(in_dim_local_palm, out_dim_local, use_bn=False)
            self.l_align_vein = FeatureAlign(in_dim_local_vein, out_dim_local, use_bn=False)

        # 局部特征的跨模态交互（CrossModalAttention）- 用于向量特征
        self.l_cross_attn = CrossModalAttention(
            dim=out_dim_local,
            num_heads=8,
            dropout=0.1
        )

        # 3) 全局特征跨模态交互（CrossModalAttention）
        self.g_cross_attn = CrossModalAttention(
            dim=out_dim_global,
            num_heads=8,
            dropout=0.1
        )

        # 4) 全局特征融合模块（通道注意力）
        self.g_fuse = PairwiseChannelAttentionFusion(dim=out_dim_global, reduction=4, dropout=0.1)

        # 5) 局部特征融合模块（通道注意力）
        self.l_fuse = PairwiseChannelAttentionFusion(dim=out_dim_local, reduction=4, dropout=0.1)

        # 6) 输出拼接 + 规范化
        # 融合后：全局维度 + 局部维度
        self.final_dim = out_dim_global + out_dim_local
        self.final_l2norm = final_l2norm

        # 7) （可选）ArcFace 头
        self.with_arcface = with_arcface
        if with_arcface:
            assert num_classes > 0, "with_arcface=True 时必须指定 num_classes"
            self.arcface = ArcMarginProduct(self.final_dim, num_classes,
                                            s=arcface_s, m=arcface_m)

    @staticmethod
    def _l2(x):
        return F.normalize(x, dim=1)

    def forward_features(self,
                         palm_global: torch.Tensor, vein_global: torch.Tensor,
                         palm_local:  torch.Tensor, vein_local:  torch.Tensor):
        """
        输入：
            palm_global, vein_global: (N, C_g) 全局向量
            palm_local, vein_local:
                - 若 use_spatial_fusion=False: (N, C_l) 向量
                - 若 use_spatial_fusion=True: (N, C_l, H, W) 空间特征图
        返回：
            fused_feat: (N, out_dim_global + out_dim_local) - 拼接后的融合向量
            details: 字典，包含中间权重/中间向量，便于可视化与消融
        """
        # ---- 1) 全局特征对齐
        g_p = self.g_align_palm(palm_global)  # (N, G)
        g_v = self.g_align_vein(vein_global)  # (N, G)

        # ---- 2) 全局特征通过跨模态注意力交互后融合
        g_p_enhanced, g_v_enhanced = self.g_cross_attn(g_p, g_v)  # (N, G), (N, G)
        # 使用通道注意力机制融合两个增强后的全局特征
        g_fused, (g_wa, g_wb) = self.g_fuse(g_p_enhanced, g_v_enhanced)  # (N, G)

        # ---- 3) 局部特征对齐与处理
        if self.use_spatial_fusion:
            # 空间特征融合路径
            l_p = self.l_align_palm(palm_local)  # (N, L, H, W)
            l_v = self.l_align_vein(vein_local)  # (N, L, H, W)
            # 池化为向量后进行跨模态交互
            l_p_vec = F.adaptive_avg_pool2d(l_p, 1).flatten(1)  # (N, L)
            l_v_vec = F.adaptive_avg_pool2d(l_v, 1).flatten(1)  # (N, L)
            l_p_enhanced, l_v_enhanced = self.l_cross_attn(l_p_vec, l_v_vec)  # (N, L), (N, L)
        else:
            # 向量特征融合路径
            l_p = self.l_align_palm(palm_local)   # (N, L)
            l_v = self.l_align_vein(vein_local)   # (N, L)
            # 局部特征跨模态交互（CrossModalAttention）
            l_p_enhanced, l_v_enhanced = self.l_cross_attn(l_p, l_v)  # (N, L), (N, L)

        # ---- 4) 局部特征通过通道注意力融合
        l_fused, (l_wa, l_wb) = self.l_fuse(l_p_enhanced, l_v_enhanced)  # (N, L)

        # ---- 5) 拼接融合后的全局特征和局部特征
        fused_feat = torch.cat([g_fused, l_fused], dim=1)  # (N, G+L)

        if self.final_l2norm:
            fused_feat = self._l2(fused_feat)

        # 便于排查/可视化
        details = {
            "global": {"palm": g_p, "vein": g_v, "palm_enhanced": g_p_enhanced,
                       "vein_enhanced": g_v_enhanced, "fused": g_fused,
                       "w_palm": g_wa, "w_vein": g_wb},
            "local":  {"palm": l_p if not self.use_spatial_fusion else l_p_vec,
                       "vein": l_v if not self.use_spatial_fusion else l_v_vec,
                       "palm_enhanced": l_p_enhanced, "vein_enhanced": l_v_enhanced,
                       "fused": l_fused, "w_palm": l_wa, "w_vein": l_wb},
            "final_fused": fused_feat
        }
        return fused_feat, details

    def forward(self,
                palm_global: torch.Tensor, vein_global: torch.Tensor,
                palm_local:  torch.Tensor, vein_local:  torch.Tensor,
                labels: torch.Tensor = None):
        fused_feat, details = self.forward_features(palm_global, vein_global,
                                                    palm_local, vein_local)
        if self.with_arcface:
            if labels is not None:
                # 训练模式：返回 logits
                logits = self.arcface(fused_feat, labels)
                return logits, fused_feat, details
            else:
                # 推理模式：只返回特征
                return fused_feat, details
        else:
            return fused_feat, details
