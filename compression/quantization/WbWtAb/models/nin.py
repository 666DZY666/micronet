import torch.nn as nn
from .util_wt_bab import Conv2d_Q

# *********************量化(三值、二值)卷积*********************
class Tnn_Bin_Conv2d(nn.Module):
    # 参数：last_relu-尾层卷积输入激活
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, last_relu=0, A=2, W=2):
        super(Tnn_Bin_Conv2d, self).__init__()
        self.A = A
        self.W = W
        self.last_relu = last_relu

        # ********************* 量化(三/二值)卷积 *********************
        self.tnn_bin_conv = Conv2d_Q(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, A=A, W=W)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.tnn_bin_conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None, A=2, W=2):
        super(Net, self).__init__()
        # 模型结构与搭建
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        self.tnn_bin = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                Tnn_Bin_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, A=A, W=W),
                Tnn_Bin_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, A=A, W=W),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Tnn_Bin_Conv2d(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, A=A, W=W),
                Tnn_Bin_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, A=A, W=W),
                Tnn_Bin_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, A=A, W=W),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Tnn_Bin_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, A=A, W=W),
                Tnn_Bin_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, last_relu=1, A=A, W=W),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.tnn_bin(x)
        x = x.view(x.size(0), -1)
        return x
