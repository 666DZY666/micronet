import torch.nn as nn
import torch
import torch.nn.functional as F

# *********************A（特征）量化（二值)***********************
class BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        output = input.sign()
        # ********************A二值——1、0*********************
        #input = torch.clamp(input, min=0)
        #print(input)
        return output, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        #*******************ste*********************
        grad_input = grad_output.clone()
        #****************saturate_ste***************
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

class TnnBinConvBNReLU(nn.Module):
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
                 dropout=0):
        super(TnnBinConvBNReLU, self).__init__()
        self.layer_type = 'TnnBinConvBNReLU'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None):
        super(Net, self).__init__()
        if cfg is None:
            # 模型结构
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        self.tnn_bin_model = nn.Sequential(
            nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(cfg[0], eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            TnnBinConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0),
            TnnBinConvBNReLU(cfg[1],  cfg[2], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            TnnBinConvBNReLU(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, dropout=0.5),
            TnnBinConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0),
            TnnBinConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            TnnBinConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, dropout=0.5),
            TnnBinConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(cfg[7], eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.tnn_bin_model(x)
        x = x.view(x.size(0), 10)
        return x
