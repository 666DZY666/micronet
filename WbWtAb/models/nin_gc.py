import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import bn

# 通道混合
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

# *********************A（特征）量化（二值)***********************
class BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        # ********************A二值——1、0*********************
        #input = torch.clamp(input, min=0)
        #print(input)
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        #*******************ste*********************
        grad_input = grad_output.clone()
        #****************saturate_ste****************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        '''
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        '''
        return grad_input

# *********************量化(三值、二值)卷积*********************
class Tnn_Bin_Conv2d(nn.Module):
    # 参数：groups-卷积分组数、channel_shuffle-通道混合标志、shuffle_groups-通道混合数（本层需与上一层分组数保持一致）、last|activation_mp|activation_nor|last_relu —— 模型结构调整标志位
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, channel_shuffle=0, shuffle_groups=1, last=0, activation_mp=0, activation_nor=1, last_relu=0, A=2):
        super(Tnn_Bin_Conv2d, self).__init__()
        self.A = A
        self.dropout_ratio = dropout
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups
        self.last = last
        self.activation_mp = activation_mp
        self.activation_nor = activation_nor
        self.last_relu = last_relu
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        #self.bn = nn.BatchNorm2d(output_channels)
        self.bn = bn.BatchNorm2d_bin(output_channels, momentum=0.1, affine_flag=2)#自定义BN_γ=1、β-train;WbAb_momentum=0.8
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.A == 2:
            if self.activation_nor:
                x, mean = BinActive()(x)
            if self.channel_shuffle_flag:
                x = channel_shuffle(x, groups=self.shuffle_groups)
            if self.dropout_ratio!=0:
                x = self.dropout(x)
            x = self.conv(x)
            x = self.bn(x)
            if self.activation_mp or self.last:
                x, mean = BinActive()(x)
            if self.last_relu:
                x = self.relu(x)
        else:
            if self.activation_nor:
                x = self.relu(x)
            if self.channel_shuffle_flag:
                x = channel_shuffle(x, groups=self.shuffle_groups)
            if self.dropout_ratio!=0:
                x = self.dropout(x)
            x = self.conv(x)
            x = self.bn(x)
            if self.activation_mp or self.last_relu:
                x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None, A=2):
        super(Net, self).__init__()
        if cfg is None:
            # 模型结构
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]

        self.tnn_bin = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                bn.BatchNorm2d_bin(cfg[0], affine_flag=2),
                Tnn_Bin_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=0, A=A),
                Tnn_Bin_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2, activation_mp=1, A=A),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                Tnn_Bin_Conv2d(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2, activation_nor=0, A=A),
                Tnn_Bin_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16, A=A),
                Tnn_Bin_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4, activation_mp=1, A=A),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                Tnn_Bin_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4, activation_nor=0, A=A),
                Tnn_Bin_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32, last=0, last_relu=1, A=A),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                bn.BatchNorm2d_bin(10, affine_flag=2),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.tnn_bin(x)
        x = x.view(x.size(0), 10)
        return x
