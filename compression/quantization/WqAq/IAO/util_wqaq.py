import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function

# ********************* range_trackers(范围统计器，统计量化前范围) *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':    # A,min_max_shape=(1, 1, 1, 1),layer级
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min_max_shape=(N, 1, 1, 1),channel级
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            
        self.update_range(min_val, max_val)
class GlobalRangeTracker(RangeTracker):  # W,min_max_shape=(N, 1, 1, 1),channel级,取本次和之前相比的min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))
class AveragedRangeTracker(RangeTracker):  # A,min_max_shape=(1, 1, 1, 1),layer级,取running_min_max —— (N, C, W, H)
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)
        
# ********************* quantizers（量化器，量化） *********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)      # 量化比例因子
        self.register_buffer('zero_point', None) # 量化零点

    def update_params(self):
        raise NotImplementedError

    # 量化
    def quantize(self, input):
        output = input * self.scale - self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # 截断
    def clamp(self, input):
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    # 反量化
    def dequantize(self, input):
        output = (input + self.zero_point) / self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.update_params()
            output = self.quantize(input)   # 量化
            output = self.round(output)
            output = self.clamp(output)     # 截断
            output = self.dequantize(output)# 反量化
        return output
class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1))
class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))
# 对称量化
class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))  # 量化后范围
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))  # 量化前范围
        self.scale = quantized_range / float_range      # 量化比例因子
        self.zero_point = torch.zeros_like(self.scale)  # 量化零点
# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        quantized_range = self.max_val - self.min_val  # 量化后范围
        float_range = self.range_tracker.max_val - self.range_tracker.min_val   # 量化前范围
        self.scale = quantized_range / float_range  # 量化比例因子
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)  # 量化零点

# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
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
        q_type=1,
        first_layer=0,
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
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        quant_input = input
        quant_weight = self.weight_quantizer(self.weight) 
        # 量化卷积
        output = F.conv2d(
            input=quant_input,
            weight=quant_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output

def reshape_to_activation(input):
  return input.reshape(1, -1, 1, 1)
def reshape_to_weight(input):
  return input.reshape(-1, 1, 1, 1)
def reshape_to_bias(input):
  return input.reshape(-1)
# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************
class QuantBNFoldConv2d(QuantConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-5,
        momentum=0.01, # 考虑量化带来的抖动影响,对momentum进行调整(0.1 ——> 0.01),削弱batch统计参数占比，一定程度抑制抖动。经实验量化训练效果更好,acc提升1%左右
        a_bits=8,
        w_bits=8,
        q_type=1,
        first_layer=0,
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
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        # 训练态
        if self.training:
            # 先做普通卷积得到A，以取得BN参数
            output = F.conv2d(
                input=input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            # 更新BN统计参数（batch和running）
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            # BN融合
            if self.bias is not None:  
              bias = reshape_to_bias(self.beta + (self.bias -  batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
              bias = reshape_to_bias(self.beta - batch_mean  * (self.gamma / torch.sqrt(batch_var + self.eps)))# b融batch
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))     # w融running
        # 测试态
        else:
            #print(self.running_mean, self.running_var)
            # BN融合
            if self.bias is not None:
              bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
              bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        
        # 量化A和bn融合后的W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        quant_input = input
        quant_weight = self.weight_quantizer(weight) 
        # 量化卷积
        if self.training:  # 训练态
          output = F.conv2d(
              input=quant_input,
              weight=quant_weight,
              bias=self.bias,  # 注意，这里不加bias（self.bias为None）
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
          # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
          output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
          output += reshape_to_activation(bias)
        else:  # 测试态
          output = F.conv2d(
              input=quant_input,
              weight=quant_weight,
              bias=bias,  # 注意，这里加bias，做完整的conv+bn
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
        return output