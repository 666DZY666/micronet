import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# A(特征)量化
class ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits  
    
    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output 

    # 量化/反量化
    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)      # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            scale = 1 / float(2 ** self.a_bits - 1)      # scale
            output = self.round(output / scale) * scale  # 量化/反量化
        return output

# W(权重)量化
class WeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits  

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output 

    # 量化/反量化
    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1                      
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5  # 归一化-[0,1]
            scale = 1 / float(2 ** self.w_bits - 1)                   # scale
            output = self.round(output / scale) * scale               # 量化/反量化
            output = 2 * output - 1
        return output


# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class QuantConv2d(nn.Conv2d):
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
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)      
    def forward(self, input):
        # 量化A和W
        quant_input = self.activation_quantizer(input)
        quant_input = input
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight) 
        else:
            quant_weight = self.weight
        # 量化卷积
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output

class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, 
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits) 

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output

# ********************* 量化全连接（同时量化A/W，并做卷积） ***********************
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=8, w_bits=8, quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)    
    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output

def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features, bias=True, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features, bias=False, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)

def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference)
    return model
    