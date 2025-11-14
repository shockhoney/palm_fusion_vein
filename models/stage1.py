import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, SqueezeExcite

# ============================================================
# ViT基础特征提取模块
# ============================================================

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:],
                      stride=self.c.stride, padding=self.c.padding,
                      dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)         
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution) 

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        return self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))

class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        self.d = int(attn_ratio * key_dim)

        qkv_out = num_heads * (2 * key_dim + self.d)
        self.qkv = Conv2d_BN(dim, qkv_out, ks=1, stride=1, pad=0)

        self.proj = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2d_BN(self.num_heads * self.d, dim, bn_weight_init=0)
        )

    def forward(self, x):  
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, self.num_heads, 2*self.key_dim + self.d, N)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.d], dim=2) 
        attn = (q.transpose(-2, -1) @ k) * self.scale 
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, self.num_heads*self.d, H, W)
        return self.proj(out)  


class EfficientViTBlock(nn.Module):
    def __init__(self, ed, kd, nh=8, ar=4, resolution=14):
        super().__init__()
        self.ffn0  = Residual(FFN(ed, int(ed * 2), resolution))
        self.mixer = Residual(GlobalSelfAttention(ed, kd, num_heads=nh, attn_ratio=ar))
        self.ffn1  = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.mixer(self.ffn0(x)))


class EfficientViT(nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=1,
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']]):
        super().__init__()

        # ---- patch embedding：四次 stride=2 下采样 → 14x14（224 输入）
        resolution = img_size
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0],       3, 2, 1, resolution=resolution // 8)
        )

        # ---- 三个 stage：仅堆叠“全局注意力 Block”，中间用 PatchMerging 做降采样
        resolution = img_size // patch_size            # 224/16 = 14
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1, self.blocks2, self.blocks3 = [], [], []

        for i, (ed, kd, dpth, nh, ar, do) in enumerate(
            zip(embed_dim, key_dim, depth, num_heads, attn_ratio, down_ops)
        ):
            # 当前分辨率下的全局注意力堆叠
            for _ in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(ed, kd, nh, ar, resolution))

            # 若需要在 stage 间降采样，用 PatchMerging 过渡
            if do[0] == 'subsample' and i < 2:   # 只在前两个 stage 后降采样
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                # 过渡：轻量残差 + 合并
                blk.append(nn.Sequential(
                    Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, resolution=resolution)),
                    Residual(FFN(ed, int(ed * 2), resolution)),
                ))
                blk.append(PatchMerging(ed, embed_dim[i+1], resolution))
                resolution = resolution_
                blk.append(nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i+1], embed_dim[i+1], 3, 1, 1,
                                        groups=embed_dim[i+1], resolution=resolution)),
                    Residual(FFN(embed_dim[i+1], int(embed_dim[i+1] * 2), resolution)),
                ))

        self.blocks1 = nn.Sequential(*self.blocks1)
        self.blocks2 = nn.Sequential(*self.blocks2)
        self.blocks3 = nn.Sequential(*self.blocks3)

    def forward_features(self, x):
        x = self.patch_embed(x)   # (N, C1, 14, 14)
        x = self.blocks1(x)       # 14x14
        x = self.blocks2(x)       # ~7x7
        x = self.blocks3(x)       # ~4x4
        return x

    def forward(self, x, pool: bool = True):
        """
        Args:
            x: 输入图像 (N, C, H, W)
            pool: 若为 True,返回全局池化后的向量；若为 False,返回空间特征图
        Returns:
            若 pool=True: (N, embed_dim[-1]) 全局向量
            若 pool=False: (N, embed_dim[-1], H, W) 空间特征图
        """
        feats = self.forward_features(x)
        if pool:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)   # (N, C)
        return feats
# ============================================================
# CNN局部细节提取模块
# ============================================================

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        # --------- 下采样层 ---------
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # --------- 主干阶段 ---------
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], 
                        drop_path=dp_rates[cur + j], 
                        layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # --------- 最后特征归一化层 ---------
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, return_spatial=False):
        """
        Args:
            x: 输入图像 (N, C, H, W)
            return_spatial: 若为 True,返回空间特征图；若为 False,返回全局平均池化后的向量
        Returns:
            若 return_spatial=True: (N, C, H', W') 空间特征图
            若 return_spatial=False: (N, C) 全局向量
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if return_spatial:
            # 返回空间特征图 (N, C, H, W)
            return x
        else:
            # 返回全局池化后的向量 (N, C)
            x = x.mean([-2, -1])  # Global Average Pooling
            x = self.norm(x)
            return x  


