import torch.nn as nn
import torch
import torch.nn.functional as F

def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class FP_Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, channel_shuffle=0, shuffle_groups=1, last=0, bin_mp=0, bin_nor=1, last_relu=0, first=0):
        super(FP_Conv2d, self).__init__()
        self.dropout_ratio = dropout
        self.last = last
        self.first_flag = first
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.first_flag:
            x = self.relu(x)
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]

        self.tnn_bin = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                FP_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, first=1, groups=2, channel_shuffle=0),
                FP_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2, bin_mp=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2, bin_nor=0),
                FP_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16),
                FP_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4, bin_mp=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4, bin_nor=0),
                FP_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )
        '''
        self.dorefa = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                FP_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, first=1, groups=2, channel_shuffle=0),
                FP_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2, bin_mp=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2, bin_nor=0),
                FP_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16),
                FP_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4, bin_mp=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4, bin_nor=0),
                FP_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )
        '''
    def forward(self, x):
        x = self.tnn_bin(x)
        #x = self.dorefa(x)
        x = x.view(x.size(0), 10)
        return x
