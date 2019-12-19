#coding=utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F

class FP_Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1):
        super(FP_Conv2d, self).__init__()
        self.dropout_ratio = dropout
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [32, 64, 128, 256, 256, 256, 512, 1024]

        self.tnn_bin = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg[0]),
                FP_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=cfg[0]),

                FP_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=cfg[1]),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=cfg[2]),
                FP_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=cfg[3]),
                FP_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=cfg[4]),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                FP_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=cfg[5]),
                FP_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=cfg[6]),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )
    def forward(self, x):
        x = self.tnn_bin(x)
        #x = self.dorefa(x)
        x = x.view(x.size(0), 10)
        return x
