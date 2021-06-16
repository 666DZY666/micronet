import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function

from micronet.base_module.op import *


# ********************* observers(统计min/max) *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':     # layer级(activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':   # channel级(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == 'FC':  # channel级(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)

class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

class HistogramObserver(nn.Module): 
    def __init__(self, q_level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.q_level = q_level
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
        self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
          # PercentileCalibrator
        max_val_cur = torch.kthvalue(input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0)[0]
          # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.max_val.copy_(max_val)


# ********************* quantizers（量化器，量化） *********************
# 取整(饱和/截断ste)
class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # 对称
        if q_type == 0:
            max_val = torch.max(torch.abs(observer_min_val), torch.abs(observer_max_val))
            min_val = -max_val
        # 非对称
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val= self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None

class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, qaft=False, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.qaft = qaft
        self.union = union
        self.q_type = 0
        # scale/zero_point/eps
        if self.observer.q_level == 'L':
            self.register_buffer('scale', torch.ones((1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((1), dtype=torch.float32))
        elif self.observer.q_level == 'C':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.q_level == 'FC':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.register_buffer('eps', torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))

    def update_qparams(self):
        raise NotImplementedError
    
    # 取整(ste)
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output
    
    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if not self.qaft:
                #qat, update quant_para
                if self.training:
                    if not self.union:
                        self.observer(input)   # update observer_min and observer_max
                    self.update_qparams()      # update scale and zero_point
            output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point,
                      self.observer.min_val / self.scale - self.zero_point,
                      self.observer.max_val / self.scale - self.zero_point, self.q_type),
                      self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale.clone()
        return output

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:   # weight
            self.register_buffer('quant_min_val', torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        elif self.activation_weight_flag == 1: # activation
            self.register_buffer('quant_min_val', torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:   # weight
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 2), dtype=torch.float32))
        elif self.activation_weight_flag == 1: # activation
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')

# 对称量化
class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = float(self.quant_max_val - self.quant_min_val) / 2                                # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))     # float_range
        scale = float_range / quant_range                                                               # scale
        scale = torch.max(scale, self.eps)                                                              # processing for very small scale
        zero_point = torch.zeros_like(scale)                                                            # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)                     # quantized_range
        float_range = self.observer.max_val - self.observer.min_val                      # float_range
        scale = float_range / quant_range                                                # scale
        scale = torch.max(scale, self.eps)                                               # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(torch.abs(self.observer.min_val / scale) + 0.5)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
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
                 q_type=0,
                 q_level=0,
                 weight_observer=0,
                 quant_inference=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
    
    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
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
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 weight_observer=0,
                 quant_inference=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                           q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                           q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)
def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)
def reshape_to_bias(input):
    return input.reshape(-1)
# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************
class QuantBNFuseConv2d(QuantConv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 eps=1e-5,
                 momentum=0.1,
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=0,
                 weight_observer=0,
                 pretrained_model=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999,
                 bn_fuse_calib=False):
        super(QuantBNFuseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                                bias, padding_mode)
        self.num_flag = 0
        self.pretrained_model = pretrained_model
        self.qaft = qaft
        self.bn_fuse_calib = bn_fuse_calib
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros((out_channels), dtype=torch.float32))
        self.register_buffer('running_var', torch.ones((out_channels), dtype=torch.float32))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                   q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                   q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                   q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                   q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='C', out_channels=out_channels), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        if not self.qaft:
            #qat, calibrate bn_statis_para
            # 训练态
            if self.training:
                # 先做普通卷积得到A，以取得BN参数
                output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                self.groups)
                # 更新BN统计参数（batch和running）
                dims = [dim for dim in range(4) if dim != 1]
                batch_mean = torch.mean(output, dim=dims)
                batch_var = torch.var(output, dim=dims)
                with torch.no_grad():
                    if not self.pretrained_model:
                        if self.num_flag == 0:
                            self.num_flag += 1
                            running_mean = batch_mean
                            running_var = batch_var
                        else:
                            running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                            running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                    else:
                        running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                        running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                # bn融合
                if self.bias is not None:
                    bias_fused = reshape_to_bias(self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
                else:
                    bias_fused = reshape_to_bias(self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))  # b融batch
                  # bn融合不校准
                if not self.bn_fuse_calib:    
                    weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(batch_var + self.eps))           # w融batch
                  # bn融合校准
                else:
                    weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))    # w融running
            # 测试态
            else:
                if self.bias is not None:
                    bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                else:
                    bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
                weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))                      # w融running
        else:
            #qaft, freeze bn_statis_para
            if self.bias is not None:
                bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
            weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))                      # w融running

        # 量化A和bn融合后的W
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(weight_fused)

        if not self.qaft:
            #qat, quant_bn_fuse_conv
            # 量化卷积
            if self.training:  # 训练态
                # bn融合不校准
                if not self.bn_fuse_calib:
                    output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                                      self.groups)
                # bn融合校准
                else:
                    output = F.conv2d(quant_input, quant_weight, None, self.stride, self.padding, self.dilation,
                                      self.groups)  # 注意，这里不加bias（self.bias为None）
                    # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
                    output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
                    output += reshape_to_activation(bias_fused)
            else:  # 测试态
                output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                                  self.groups)  # 注意，这里加bias，做完整的conv+bn
        else:
            #qaft, quant_bn_fuse_conv
            output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                              self.groups)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=0,
                 weight_observer=0,
                 quant_inference=False,
                 qaft=False,
                 ptq=False,
                 percentile=0.9999):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                   q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                   q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                   q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                   q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
                if weight_observer == 0:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
                else:
                    if q_level == 0:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                    else:
                        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                    q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='FC', out_channels=out_features), activation_weight_flag=0, qaft=qaft)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=0, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantReLU, self).__init__(inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output


class QuantLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False, a_bits=8, q_type=0, qaft=False,
                 ptq=False, percentile=0.9999):
        super(QuantLeakyReLU, self).__init__(negative_slope, inplace)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.leaky_relu(quant_input, self.negative_slope, self.inplace)
        return output


class QuantSigmoid(nn.Sigmoid):
    def __init__(self, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantSigmoid, self).__init__()
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.sigmoid(quant_input)
        return output


class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.max_pool2d(quant_input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.return_indices, self.ceil_mode)
        return output


class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(quant_input, self.kernel_size, self.stride, self.padding,
                              self.ceil_mode, self.count_include_pad, self.divisor_override)
        return output


class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        if not ptq:
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft)
        else:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(quant_input, self.output_size)
        return output


class QuantAdd(nn.Module):
    def __init__(self, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantAdd, self).__init__()
        if not ptq:
            self.observer_res = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            self.observer_shortcut = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
        else:
            self.observer_res = HistogramObserver(q_level='L', percentile=percentile)
            self.observer_shortcut = HistogramObserver(q_level='L', percentile=percentile)
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft, union=True)

    def forward(self, res, shortcut):
        self.observer_res(res)
        self.observer_shortcut(shortcut)
        observer_min_val = torch.min(self.observer_res.min_val, self.observer_shortcut.min_val)
        observer_max_val = torch.max(self.observer_res.max_val, self.observer_shortcut.max_val)
        self.activation_quantizer.observer.min_val = observer_min_val
        self.activation_quantizer.observer.max_val = observer_max_val
        quant_res = self.activation_quantizer(res)
        quant_shortcut = self.activation_quantizer(shortcut)
        output = quant_res + quant_shortcut
        return output


def add_quant_op(module, a_bits=8, w_bits=8, q_type=0, q_level=0, weight_observer=0,
                 bn_fuse=False, bn_fuse_calib=False, quant_inference=False,
                 pretrained_model=False, qaft=False, ptq=False, percentile=0.9999):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if bn_fuse:
                conv_name_temp = name
                conv_child_temp = child
            else:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, q_type=q_type,
                                             q_level=q_level, weight_observer=weight_observer,
                                             quant_inference=quant_inference, qaft=qaft, ptq=ptq, percentile=percentile)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, q_type=q_type,
                                             q_level=q_level, weight_observer=weight_observer,
                                             quant_inference=quant_inference, qaft=qaft, ptq=ptq, percentile=percentile)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.BatchNorm2d):
            if bn_fuse:
                if conv_child_temp.bias is not None:
                    quant_bn_fuse_conv = QuantBNFuseConv2d(conv_child_temp.in_channels,
                                                           conv_child_temp.out_channels,
                                                           conv_child_temp.kernel_size,
                                                           stride=conv_child_temp.stride,
                                                           padding=conv_child_temp.padding,
                                                           dilation=conv_child_temp.dilation,
                                                           groups=conv_child_temp.groups,
                                                           bias=True,
                                                           padding_mode=conv_child_temp.padding_mode,
                                                           eps=child.eps,
                                                           momentum=child.momentum,
                                                           a_bits=a_bits,
                                                           w_bits=w_bits,
                                                           q_type=q_type,
                                                           q_level=q_level,
                                                           weight_observer=weight_observer,
                                                           pretrained_model=pretrained_model,
                                                           qaft=qaft,
                                                           ptq=ptq,
                                                           percentile=percentile,
                                                           bn_fuse_calib=bn_fuse_calib)
                    quant_bn_fuse_conv.bias.data = conv_child_temp.bias
                else:
                    quant_bn_fuse_conv = QuantBNFuseConv2d(conv_child_temp.in_channels,
                                                           conv_child_temp.out_channels,
                                                           conv_child_temp.kernel_size,
                                                           stride=conv_child_temp.stride,
                                                           padding=conv_child_temp.padding,
                                                           dilation=conv_child_temp.dilation,
                                                           groups=conv_child_temp.groups,
                                                           bias=False,
                                                           padding_mode=conv_child_temp.padding_mode,
                                                           eps=child.eps,
                                                           momentum=child.momentum,
                                                           a_bits=a_bits,
                                                           w_bits=w_bits,
                                                           q_type=q_type,
                                                           q_level=q_level,
                                                           weight_observer=weight_observer,
                                                           pretrained_model=pretrained_model,
                                                           qaft=qaft,
                                                           ptq=ptq,
                                                           percentile=percentile,
                                                           bn_fuse_calib=bn_fuse_calib)
                quant_bn_fuse_conv.weight.data = conv_child_temp.weight
                quant_bn_fuse_conv.gamma.data = child.weight
                quant_bn_fuse_conv.beta.data = child.bias
                quant_bn_fuse_conv.running_mean.copy_(child.running_mean)
                quant_bn_fuse_conv.running_var.copy_(child.running_var)
                module._modules[conv_name_temp] = quant_bn_fuse_conv
                module._modules[name] = nn.Identity()
        elif isinstance(child, nn.ConvTranspose2d):
            if child.bias is not None:
                quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                            child.out_channels,
                                                            child.kernel_size,
                                                            stride=child.stride,
                                                            padding=child.padding,
                                                            output_padding=child.output_padding,
                                                            groups=child.groups,
                                                            bias=True,
                                                            dilation=child.dilation,
                                                            padding_mode=child.padding_mode,
                                                            a_bits=a_bits,
                                                            w_bits=w_bits,
                                                            q_type=q_type,
                                                            weight_observer=weight_observer,
                                                            quant_inference=quant_inference,
                                                            qaft=qaft,
                                                            ptq=ptq,
                                                            percentile=percentile)
                quant_conv_transpose.bias.data = child.bias
            else:
                quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                            child.out_channels,
                                                            child.kernel_size,
                                                            stride=child.stride,
                                                            padding=child.padding,
                                                            output_padding=child.output_padding,
                                                            groups=child.groups,
                                                            bias=False,
                                                            dilation=child.dilation,
                                                            padding_mode=child.padding_mode,
                                                            a_bits=a_bits,
                                                            w_bits=w_bits,
                                                            q_type=q_type,
                                                            weight_observer=weight_observer,
                                                            quant_inference=quant_inference,
                                                            qaft=qaft,
                                                            ptq=ptq,
                                                            percentile=percentile)
            quant_conv_transpose.weight.data = child.weight
            module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            if child.bias is not None:
                quant_linear = QuantLinear(child.in_features, child.out_features,
                                           bias=True, a_bits=a_bits, w_bits=w_bits,
                                           q_type=q_type, q_level=q_level,
                                           weight_observer=weight_observer,
                                           quant_inference=quant_inference,
                                           qaft=qaft,
                                           ptq=ptq,
                                           percentile=percentile)
                quant_linear.bias.data = child.bias
            else:
                quant_linear = QuantLinear(child.in_features, child.out_features,
                                           bias=False, a_bits=a_bits, w_bits=w_bits,
                                           q_type=q_type, q_level=q_level,
                                           weight_observer=weight_observer,
                                           quant_inference=quant_inference,
                                           qaft=qaft,
                                           ptq=ptq,
                                           percentile=percentile)
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear
        # relu needn’t quantize, it will be fused in quant_inference
        #elif isinstance(child, nn.ReLU):
        #    quant_relu = QuantReLU(inplace=child.inplace, a_bits=a_bits,
        #                           q_type=q_type, qaft=qaft, ptq=ptq, percentile=percentile)
        #    module._modules[name] = quant_relu
        elif isinstance(child, nn.LeakyReLU):
            quant_leaky_relu = QuantLeakyReLU(negative_slope=child.negative_slope,
                                              inplace=child.inplace,
                                              a_bits=a_bits,
                                              q_type=q_type,
                                              qaft=qaft,
                                              ptq=ptq,
                                              percentile=percentile)
            module._modules[name] = quant_leaky_relu
        elif isinstance(child, nn.Sigmoid):
            quant_sigmoid = QuantSigmoid(a_bits=a_bits,
                                         q_type=q_type,
                                         qaft=qaft,
                                         ptq=ptq,
                                         percentile=percentile)
            module._modules[name] = quant_sigmoid
        elif isinstance(child, nn.MaxPool2d):
            quant_max_pool = QuantMaxPool2d(kernel_size=child.kernel_size,
                                            stride=child.stride,
                                            padding=child.padding,
                                            a_bits=a_bits,
                                            q_type=q_type,
                                            qaft=qaft,
                                            ptq=ptq,
                                            percentile=percentile)
            module._modules[name] = quant_max_pool
        elif isinstance(child, nn.AvgPool2d):
            quant_avg_pool = QuantAvgPool2d(kernel_size=child.kernel_size,
                                            stride=child.stride,
                                            padding=child.padding,
                                            a_bits=a_bits,
                                            q_type=q_type,
                                            qaft=qaft,
                                            ptq=ptq,
                                            percentile=percentile)
            module._modules[name] = quant_avg_pool
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            quant_adaptive_avg_pool = QuantAdaptiveAvgPool2d(output_size=child.output_size,
                                                             a_bits=a_bits,
                                                             q_type=q_type,
                                                             qaft=qaft,
                                                             ptq=ptq,
                                                             percentile=percentile)
            module._modules[name] = quant_adaptive_avg_pool
        elif isinstance(child, Add):
            quant_add = QuantAdd(a_bits=a_bits,
                                 q_type=q_type,
                                 qaft=qaft,
                                 ptq=ptq,
                                 percentile=percentile)
            module._modules[name] = quant_add
        #elif isinstance(child, Concat):
        #    quant_concat = QuantConcat(dim=child.dim,
        #                               a_bits=a_bits,
        #                               q_type=q_type,
        #                               qaft=qaft,
        #                               ptq=ptq,
        #                               percentile=percentile)
        #    module._modules[name] = quant_concat
        else:
            add_quant_op(child, a_bits=a_bits, w_bits=w_bits, q_type=q_type,
                         q_level=q_level, weight_observer=weight_observer,
                         bn_fuse=bn_fuse, bn_fuse_calib=bn_fuse_calib,
                         quant_inference=quant_inference,
                         pretrained_model=pretrained_model,
                         qaft=qaft, ptq=ptq, percentile=percentile)


def prepare(model, inplace=False, a_bits=8, w_bits=8, q_type=0, q_level=0,
            weight_observer=0, bn_fuse=False, bn_fuse_calib=False,
            quant_inference=False, pretrained_model=False, qaft=False,
            ptq=False, percentile=0.9999):
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(model, a_bits=a_bits, w_bits=w_bits, q_type=q_type,
                 q_level=q_level, weight_observer=weight_observer,
                 bn_fuse=bn_fuse, bn_fuse_calib=bn_fuse_calib,
                 quant_inference=quant_inference,
                 pretrained_model=pretrained_model,
                 qaft=qaft, ptq=ptq, percentile=percentile)
    return model


# *** temp_dev ***
'''
class QuantConcat(nn.Module):
    def __init__(self, dim=1, a_bits=8, q_type=0, qaft=False, ptq=False, percentile=0.9999):
        super(QuantConcat, self).__init__()
        self.dim = dim
        if not ptq:
            self.observer_res = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            self.observer_shortcut = MovingAverageMinMaxObserver(q_level='L', out_channels=None)
            if q_type == 0:
                self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                               q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
            else:
                self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=1, qaft=qaft, union=True)
        else:
            self.observer_res = HistogramObserver(q_level='L', percentile=percentile)
            self.observer_shortcut = HistogramObserver(q_level='L', percentile=percentile)
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(
                                                           q_level='L', percentile=percentile), activation_weight_flag=1, qaft=qaft, union=True)

    def forward(self, res, shortcut):
        self.observer_res(res)
        self.observer_shortcut(shortcut)
        observer_min_val = torch.min(self.observer_res.min_val, self.observer_shortcut.min_val)
        observer_max_val = torch.max(self.observer_res.max_val, self.observer_shortcut.max_val)
        self.activation_quantizer.observer.min_val = observer_min_val
        self.activation_quantizer.observer.max_val = observer_max_val
        quant_res = self.activation_quantizer(res)
        quant_shortcut = self.activation_quantizer(shortcut)
        output = torch.cat([quant_shortcut, quant_res], dim=self.dim)
        return output
'''
