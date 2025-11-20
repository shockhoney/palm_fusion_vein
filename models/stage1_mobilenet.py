from torch.nn import Conv2d, BatchNorm1d, BatchNorm2d, Sequential, Module, LeakyReLU

class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        # self.prelu = PReLU(out_c)
        self.relu = LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.prelu(x)
        x = self.relu(x)
        return x

class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DepthWise(Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.conv = ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return x

class DepthWiseResidual(Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWiseResidual, self).__init__()
        self.conv = ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        output = short_cut + x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWiseResidual(c, c, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, input_channel=1, input_width=224, input_height=224):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(input_channel, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
        self.conv_23 = DepthWise(32, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=64)
        self.conv_3 = Residual(32, num_block=4, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = DepthWise(32, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_4 = Residual(64, num_block=6, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_5 = Residual(64, num_block=2, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = ConvBlock(64, 256, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(256, 256, groups=256, kernel=(int(input_height / 16), int(input_width / 16)), stride=(1, 1), padding=(0, 0))

        self.bn = BatchNorm1d(256)

    def forward(self, x, return_spatial=False):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        spatial_feat = self.conv_6_sep(x)
        global_feat = self.conv_6_dw(spatial_feat)
        global_feat = global_feat.reshape(-1, 256)
        global_feat = self.bn(global_feat)

        if return_spatial:
            return spatial_feat

        return global_feat
