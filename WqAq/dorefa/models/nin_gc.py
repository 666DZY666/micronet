import torch
import torch.nn as nn
import torch.nn.functional as F
from util_wqaq import conv2d_Q_fn, activation_quantize_fn

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
            kernel_size=-1, stride=-1, padding=-1, dropout=0, wbits=8, abits=8, quan_mp=0, quan_nor=1, last=0, last_relu=0, groups=1, channel_shuffle=0, shuffle_groups=1):
        super(DorefaConv2d, self).__init__()
        self.dropout_ratio = dropout
        self.quan_mp = quan_mp
        self.quan_nor = quan_nor
        self.last = last
        self.last_relu = last_relu
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        Conv2d = conv2d_Q_fn(w_bit=wbits)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.act_q = activation_quantize_fn(a_bit=abits)

    def forward(self, x):
        if self.quan_nor:
            x = self.act_q(self.relu(x))
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.quan_mp or self.last:
            x = self.act_q(self.relu(x))
        if self.last_relu:
            x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None, wbits=8, abits=8):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]

        self.dorefa = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                DorefaConv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, wbits=wbits, abits=abits, groups=2, channel_shuffle=0),
                DorefaConv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, quan_mp=1, wbits=wbits, abits=abits, groups=2, channel_shuffle=1, shuffle_groups=2),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                
                DorefaConv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, quan_nor=0, wbits=wbits, abits=abits, groups=16, channel_shuffle=1, shuffle_groups=2),
                DorefaConv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, wbits=wbits, abits=abits, groups=4, channel_shuffle=1, shuffle_groups=16),
                DorefaConv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, quan_mp=1, wbits=wbits, abits=abits, groups=4, channel_shuffle=1, shuffle_groups=4),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                
                DorefaConv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, quan_nor=0, wbits=wbits, abits=abits, groups=32, channel_shuffle=1, shuffle_groups=4),
                DorefaConv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, wbits=wbits, abits=abits, groups=8, channel_shuffle=1, shuffle_groups=32, last=0, last_relu=1),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.dorefa(x)
        x = x.view(x.size(0), 10)
        return x