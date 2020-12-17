import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Round(Function):
    
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# ********************* A(特征)量化 ***********************
class ActivationQuantize(nn.Module):
  def __init__(self, a_bits):
    super().__init__()
    self.a_bits = a_bits

  def round(self, input):
    output = Round.apply(input)
    return output

  def forward(self, input):
    if self.a_bits == 32:
      output = input
    elif self.a_bits == 1:
      print('！Binary quantization is not supported ！')
      assert self.a_bits != 1
    else:
      output = torch.clamp(input * 0.1, 0, 1)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
      scale = float(2 ** self.a_bits - 1)
      output = output * scale
      output = self.round(output)
      output = output / scale
    return output
# ********************* W(模型参数)量化 ***********************
class WeightQuantize(nn.Module):
  def __init__(self, w_bits):
    super().__init__()
    self.w_bits = w_bits

  def round(self, input):
    output = Round.apply(input)
    return output

  def forward(self, input):
    if self.w_bits == 32:
      output = input
    elif self.w_bits == 1:
      print('！Binary quantization is not supported ！')
      assert self.w_bits != 1                      
    else:
      output = torch.tanh(input)
      output = output / 2 / torch.max(torch.abs(output)) + 0.5  #归一化-[0,1]
      scale = float(2 ** self.w_bits - 1)
      output = output * scale
      output = self.round(output)
      output = output / scale
      output = 2 * output - 1
    return output

# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class QuantConv2d(nn.Conv2d):
  def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        a_bits=8,
        w_bits=8,
        first_layer=0
      ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationQuantize(a_bits=a_bits)
        self.weight_quantizer = WeightQuantize(w_bits=w_bits)    
        self.first_layer = first_layer

  def forward(self, input):
    # 量化A和W
    if not self.first_layer:
      input = self.activation_quantizer(input)
    q_input = input
    q_weight = self.weight_quantizer(self.weight) 
    # 量化卷积
    output = F.conv2d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    return output
# ********************* 量化全连接（同时量化A/W，并做全连接） ***********************
class QuantLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
    super().__init__(in_features=in_features, out_features=out_features, bias=bias)
    self.activation_quantizer = ActivationQuantize(a_bits=a_bits)
    self.weight_quantizer = WeightQuantize(w_bits=w_bits) 

  def forward(self, input):
    # 量化A和W
    q_input = self.activation_quantizer(input)
    q_weight = self.weight_quantizer(self.weight) 
    # 量化全连接
    output = F.linear(input=q_input, weight=q_weight, bias=self.bias)
    return output