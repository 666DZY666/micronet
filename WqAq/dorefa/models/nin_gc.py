import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_wqaq import Conv2d_Q

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

class DorefaConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, channel_shuffle=0, shuffle_groups=1, last_relu=0, abits=8, wbits=8, first_layer=0):
        super(DorefaConv2d, self).__init__()
        self.last_relu = last_relu
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.first_layer = first_layer

        self.q_conv = Conv2d_Q(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, first_layer=first_layer)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.first_layer:
            x = self.relu(x)
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.q_conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None, abits=8, wbits=8):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]

        # model - A/W全量化(除输入、输出外)
        self.dorefa = nn.Sequential(
                DorefaConv2d(3, cfg[0], kernel_size=5, stride=1, padding=2, abits=abits, wbits=wbits, first_layer=1),
                DorefaConv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=0, abits=abits, wbits=wbits),
                DorefaConv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2, abits=abits, wbits=wbits),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                
                DorefaConv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2, abits=abits, wbits=wbits),
                DorefaConv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16, abits=abits, wbits=wbits),
                DorefaConv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4, abits=abits, wbits=wbits),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                
                DorefaConv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4, abits=abits, wbits=wbits),
                DorefaConv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32, abits=abits, wbits=wbits),
                DorefaConv2d(cfg[7], 10, kernel_size=1, stride=1, padding=0, last_relu=1, abits=abits, wbits=wbits),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.dorefa(x)
        x = x.view(x.size(0), -1)
        return x