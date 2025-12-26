
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return x


class DepthWiseResidual(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return short_cut + x


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.model = nn.Sequential(*[
            DepthWiseResidual(c, c, kernel=kernel, padding=padding, stride=stride, groups=groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.model(x)


class SmallMobileFaceNet(nn.Module):
    """
    ~0.45x of your teacher (depends on embedding_size). Keep same stride pattern and embedding_size.
    """
    def __init__(self, input_channel=1, embedding_size=256):
        super().__init__()
        self.conv1 = ConvBlock(input_channel, 24, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(24, 24, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=24)
        self.conv_23 = DepthWise(24, 24, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=48)
        self.conv_3 = Residual(24, num_block=3, groups=48)
        self.conv_34 = DepthWise(24, 48, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=96)
        self.conv_4 = Residual(48, num_block=4, groups=96)
        self.conv_45 = DepthWise(48, 48, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=192)
        self.conv_5 = Residual(48, num_block=1, groups=96)
        self.conv_6_sep = ConvBlock(48, embedding_size, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)

        self.out_dim = embedding_size
        self.local_dim = embedding_size

    def forward(self, x, return_spatial=False):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        feat_map = self.conv_6_sep(x)

        emb = self.global_pool(feat_map).flatten(1)
        emb = self.bn(emb)
        return feat_map if return_spatial else emb


class TinyMobileFaceNet(nn.Module):
    """
    Very tiny version (~0.2x of your teacher). Useful when you want extreme lightweight.
    """
    def __init__(self, input_channel=1, embedding_size=256):
        super().__init__()
        self.conv1 = ConvBlock(input_channel, 16, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(16, 16, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
        self.conv_23 = DepthWise(16, 16, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=32)
        self.conv_3 = Residual(16, num_block=2, groups=32)
        self.conv_34 = DepthWise(16, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=64)
        self.conv_4 = Residual(32, num_block=3, groups=64)
        self.conv_45 = DepthWise(32, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_5 = Residual(32, num_block=1, groups=64)
        self.conv_6_sep = ConvBlock(32, embedding_size, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(embedding_size)

        self.out_dim = embedding_size
        self.local_dim = embedding_size

    def forward(self, x, return_spatial=False):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        feat_map = self.conv_6_sep(x)

        emb = self.global_pool(feat_map).flatten(1)
        emb = self.bn(emb)
        return feat_map if return_spatial else emb



