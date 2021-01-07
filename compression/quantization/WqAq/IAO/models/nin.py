import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_wqaq import QuantConv2d, QuantBNFuseConv2d, QuantReLU, QuantMaxPool2d, QuantAvgPool2d

class QuantConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 eps=1e-5,
                 momentum=0.1,
                 a_bits=8,
                 w_bits=8,
                 bn_fuse=0,
                 q_type=1,
                 q_level=0,
                 first_layer=0):
        super(QuantConvBNReLU, self).__init__()
        self.bn_fuse = bn_fuse

        if self.bn_fuse == 1:
            self.quant_bn_fuse_conv = QuantBNFuseConv2d(in_channels, out_channels,
                                                        kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, eps=eps, momentum=momentum, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, first_layer=first_layer)
        else:
            self.quant_conv = QuantConv2d(in_channels, out_channels,
                                          kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, first_layer=first_layer)
            self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.relu = QuantReLU(inplace=True, a_bits=a_bits, q_type=q_type)

    def forward(self, x):
        if self.bn_fuse == 1:
            x = self.quant_bn_fuse_conv(x)
        else:
            x = self.quant_conv(x)
            x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None, a_bits=8, w_bits=8, bn_fuse=0, q_type=1, q_level=0):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        # model - A/W全量化(除输入、输出外)
        self.quant_model = nn.Sequential(
            QuantConvBNReLU(3, cfg[0], kernel_size=5, stride=1, padding=2, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level, first_layer=1),
            QuantConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantMaxPool2d(kernel_size=3, stride=2, padding=1, a_bits=a_bits, q_type=q_type),

            QuantConvBNReLU(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantMaxPool2d(kernel_size=3, stride=2, padding=1, a_bits=a_bits, q_type=q_type),

            QuantConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantConvBNReLU(cfg[7], 10, kernel_size=1, stride=1, padding=0, a_bits=a_bits, w_bits=w_bits, bn_fuse=bn_fuse, q_type=q_type, q_level=q_level),
            QuantAvgPool2d(kernel_size=8, stride=1, padding=0, a_bits=a_bits, q_type=q_type)
        )

    def forward(self, x):
        x = self.quant_model(x)
        x = x.view(x.size(0), -1)
        return x
