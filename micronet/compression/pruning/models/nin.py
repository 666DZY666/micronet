import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
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
                 momentum=0.1,
                 quant_type=0,
                 first_relu=0):
        super(ConvBNReLU, self).__init__()
        self.quant_type = quant_type
        self.first_relu = first_relu
        if self.quant_type == 0:
            self.tnn_bin_conv = nn.Conv2d(in_channels, out_channels,
                                          kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        elif self.quant_type == 1:
            self.quant_conv = nn.Conv2d(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.first_relu == 1:
            x = self.relu(x)
        if self.quant_type == 0:
            x = self.tnn_bin_conv(x)
        elif self.quant_type == 1:
            x = self.quant_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg=None, quant_type=0):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        self.quant_type = quant_type 
        if self.quant_type == 0:
            self.tnn_bin_model = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                ConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, first_relu=1, quant_type=quant_type),
                ConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ConvBNReLU(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, quant_type=quant_type),
                ConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                ConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, quant_type=quant_type),
                ConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            )
        elif self.quant_type == 1:
            self.quant_model = nn.Sequential(
                ConvBNReLU(3, cfg[0], kernel_size=5, stride=1, padding=2, quant_type=quant_type),
                ConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                ConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ConvBNReLU(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, quant_type=quant_type),
                ConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                ConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                ConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, quant_type=quant_type),
                ConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                ConvBNReLU(cfg[7],  10, kernel_size=1, stride=1, padding=0, quant_type=quant_type),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            )

    def forward(self, x):
        if self.quant_type == 0:
            x = self.tnn_bin_model(x)
        elif self.quant_type == 1:
            x = self.quant_model(x)
        x = x.view(x.size(0), -1)
        return x
