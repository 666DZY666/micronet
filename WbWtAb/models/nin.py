#coding=utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F
# 导入自定义的bn层
from layers import bn

# *********************A（特征）量化（二值)***********************
# 手动扩展op。
class BinActive(torch.autograd.Function):

    def forward(self, input):
        # 用 self 把该存的存起来，留着 backward的时候用
        self.save_for_backward(input)
        # 输入尺寸
        size = input.size()
        # 求所有权值的绝对值的均值
        mean = torch.mean(input.abs(), 1, keepdim=True)
        # 记录输入的符号
        input = input.sign()
        # ********************A二值——1、0*********************
        #input = torch.clamp(input, min=0)
        #print(input)
        # 返回input作为输出，即二值网络只用权值的符号来模拟原始权重
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        # 在开头的地方将保存的 tensor 给 unpack 了
        input, = self.saved_tensors
        #*******************ste*********************
        grad_input = grad_output.clone()
        #****************saturate_ste***************
        # torch.ge(input, other, out=None)表示对比每一个input和other是否有如下关系
        # input≥otherinput≥other，input和other均为tensor，输出为一个二值tensor。
        # torch.le表示<=关系
        # 这里的目的是把梯度限制在[-1,1]之间
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
        # 返回梯度
        return grad_input

# *********************量化(三值、二值)卷积*********************
class Tnn_Bin_Conv2d(nn.Module):
    # 参数：last|activation_mp|activation_nor|last_relu —— 模型结构调整标志位
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, last=0, activation_mp=0, activation_nor=1, last_relu=0, A=2):
        super(Tnn_Bin_Conv2d, self).__init__()
        self.A = A
        self.dropout_ratio = dropout
        self.last = last
        self.activation_mp = activation_mp
        self.activation_nor = activation_nor
        self.last_relu = last_relu
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn = nn.BatchNorm2d(output_channels)
        self.bn = bn.BatchNorm2d_bin(output_channels, momentum=0.1, affine_flag=2)#自定义BN_γ=1、β-train;WbAb_momentum=0.8
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.A == 2:
            if self.activation_nor :
                x, mean = BinActive()(x)
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
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]

        self.tnn_bin = nn.Sequential(
                nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                bn.BatchNorm2d_bin(cfg[0], affine_flag=2),
                Tnn_Bin_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, A=A),
                Tnn_Bin_Conv2d(cfg[1],  cfg[2], kernel_size=1, stride=1, padding=0, activation_mp=1, A=A),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Tnn_Bin_Conv2d(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, activation_nor=0, A=A),
                Tnn_Bin_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, A=A),
                Tnn_Bin_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, activation_mp=1, A=A),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                Tnn_Bin_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, activation_nor=0, A=A),
                Tnn_Bin_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, last=0, last_relu=1, A=A),
                nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                bn.BatchNorm2d_bin(10, affine_flag=2),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.tnn_bin(x)
        x = x.view(x.size(0), 10)
        return x
