import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* observers(统计min/max) *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level, device):
        super(ObserverBase, self).__init__()
        self.q_level = q_level
        self.device = device

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
    def __init__(self, q_level, device, out_channels):
        super(MinMaxObserver, self).__init__(q_level, device)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':            
            self.min_val = torch.zeros((1), dtype=torch.float32, device=self.device)
            self.max_val = torch.zeros((1), dtype=torch.float32, device=self.device)
        elif self.q_level == 'C':          
            self.min_val = torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32, device=self.device)
            self.max_val =  torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32, device=self.device)
        elif self.q_level == 'FC':         
            self.min_val = torch.zeros((out_channels, 1), dtype=torch.float32, device=self.device)
            self.max_val = torch.zeros((out_channels, 1), dtype=torch.float32, device=self.device)

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
    def __init__(self, q_level, device, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level, device)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':            
            self.min_val = torch.zeros((1), dtype=torch.float32, device=self.device)
            self.max_val = torch.zeros((1), dtype=torch.float32, device=self.device)
        elif self.q_level == 'C':          
            self.min_val = torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32, device=self.device)
            self.max_val =  torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32, device=self.device)
        elif self.q_level == 'FC':         
            self.min_val = torch.zeros((out_channels, 1), dtype=torch.float32, device=self.device)
            self.max_val = torch.zeros((out_channels, 1), dtype=torch.float32, device=self.device)

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

class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        #scale/zero_point/eps
        if self.observer.q_level == 'L':            
            self.register_buffer('scale', torch.ones((1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((1), dtype=torch.float32))
        elif self.observer.q_level == 'C':      
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.q_level == 'FC':         
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.eps = torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32, device=self.observer.device) # eps(1.1921e-07)

    def update_qparams(self):
        raise NotImplementedError

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if self.training:
                self.observer(input)
                self.update_qparams()  # update scale and zero_point
            # 量化/反量化
            output = (torch.clamp(self.round(input / self.scale - self.zero_point), self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale
        return output

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        self.quant_min_val = torch.tensor((-(1 << (self.bits - 1))), device=self.observer.device)
        self.quant_max_val = torch.tensor(((1 << (self.bits - 1)) - 1), device=self.observer.device)

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        self.quant_min_val = torch.tensor((0), device=self.observer.device)
        self.quant_max_val = torch.tensor(((1 << self.bits) - 1), device=self.observer.device)

# 对称量化
class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        # quantized_range
        if self.activation_weight_flag == 0: 
            quant_range = float(torch.min(torch.abs(self.quant_min_val), torch.abs(self.quant_max_val)))    # weight               
        else:                              
            quant_range = float(self.quant_max_val - self.quant_min_val) / 2                                # activation
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))     # float_range
        self.scale = float_range / quant_range                                                          # scale
        self.scale = torch.max(self.scale, self.eps)                                                    # processing for very small scale
        self.zero_point = torch.zeros_like(self.scale)                                                  # zero_point

# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_qparams(self):
        quant_range = float(self.quant_max_val - self.quant_min_val)           # quantized_range
        float_range = self.observer.max_val - self.observer.min_val            # float_range
        self.scale = float_range / quant_range                                 # scale
        self.scale = torch.max(self.scale, self.eps)                           # processing for very small scale
        self.zero_point = torch.round(self.observer.min_val / self.scale)      # zero_point


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
                 device='cpu',
                 weight_observer=0,
                 quant_inference=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            else:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        # 量化A和W
        quant_input = self.activation_quantizer(input)
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
                 q_type=0,
                 device='cpu',
                 weight_observer=0,
                 quant_inference=False):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, 
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
            if weight_observer == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            if weight_observer == 0:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            else:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

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
                 device='cpu',
                 weight_observer=0):
        super(QuantBNFuseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                                bias, padding_mode)
        self.num_flag = 0
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros((out_channels), dtype=torch.float32))
        self.register_buffer('running_var', torch.ones((out_channels), dtype=torch.float32))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            else:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='C', out_channels=out_channels, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
       
    def forward(self, input):
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
                if self.num_flag == 0:
                    self.num_flag += 1
                    running_mean = batch_mean
                    running_var = batch_var
                else:
                    running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.running_mean.copy_(running_mean)
                self.running_var.copy_(running_var)
            # BN融合
            if self.bias is not None:  
              bias_fused = reshape_to_bias(self.beta + (self.bias -  batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
              bias_fused = reshape_to_bias(self.beta - batch_mean  * (self.gamma / torch.sqrt(batch_var + self.eps)))# b融batch
            weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))     # w融running
        # 测试态
        else:
            # BN融合
            if self.bias is not None:
              bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
              bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
            weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
        
        # 量化A和bn融合后的W
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(weight_fused) 
        # 量化卷积
        if self.training:  # 训练态
          output = F.conv2d(quant_input, quant_weight, None, self.stride, self.padding, self.dilation,
                            self.groups) # 注意，这里不加bias（self.bias为None）
          # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
          output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
          output += reshape_to_activation(bias_fused)
        else:  # 测试态
          output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                            self.groups) # 注意，这里加bias，做完整的conv+bn
        return output

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=8, w_bits=8, q_type=0, q_level=0, device='cpu', weight_observer=0, quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='FC', out_channels=out_features, device=device), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
            else:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='FC', out_channels=out_features, device=device), activation_weight_flag=2)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)
    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output
        
class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, a_bits=8, q_type=0, device='cpu'):
        super(QuantReLU, self).__init__(inplace)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output

class QuantSigmoid(nn.Sigmoid):
    def __init__(self, a_bits=8, q_type=0, device='cpu'):
        super(QuantSigmoid, self).__init__()
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.sigmoid(quant_input)
        return output

class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, a_bits=8, q_type=0, device='cpu'):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.max_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)
        return output
        
class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, a_bits=8, q_type=0, device='cpu'):
        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        return output
        
class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, a_bits=8, q_type=0, device='cpu'):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(quant_input, self.output_size)
        return output

def add_quant_op(module, a_bits=8, w_bits=8, q_type=0, q_level=0, device='cpu', weight_observer=0, bn_fuse=0, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if bn_fuse:
                conv_name_temp = name
                conv_child_temp = child
            else:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.BatchNorm2d):
            if bn_fuse:
                if conv_child_temp.bias is not None:
                    quant_bn_fuse_conv = QuantBNFuseConv2d(conv_child_temp.in_channels, conv_child_temp.out_channels,
                                                           conv_child_temp.kernel_size, stride=conv_child_temp.stride, padding=conv_child_temp.padding, dilation=conv_child_temp.dilation, groups=conv_child_temp.groups, bias=True, padding_mode=conv_child_temp.padding_mode, eps=child.eps, momentum=child.momentum, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer)
                    quant_bn_fuse_conv.bias.data = conv_child_temp.bias
                else:
                    quant_bn_fuse_conv = QuantBNFuseConv2d(conv_child_temp.in_channels, conv_child_temp.out_channels,
                                                           conv_child_temp.kernel_size, stride=conv_child_temp.stride, padding=conv_child_temp.padding, dilation=conv_child_temp.dilation, groups=conv_child_temp.groups, bias=False, padding_mode=conv_child_temp.padding_mode, eps=child.eps, momentum=child.momentum, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer)
                quant_bn_fuse_conv.weight.data = conv_child_temp.weight
                quant_bn_fuse_conv.gamma.data = child.weight
                quant_bn_fuse_conv.beta.data = child.bias
                quant_bn_fuse_conv.running_mean.copy_(child.running_mean)
                quant_bn_fuse_conv.running_var.copy_(child.running_var)
                quant_bn_fuse_conv.eps = child.eps
                module._modules[conv_name_temp] = quant_bn_fuse_conv
                module._modules[name] = nn.Identity()
        elif isinstance(child, nn.ConvTranspose2d):
            if child.bias is not None:
                quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                            child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, q_type=q_type, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
                quant_conv_transpose.bias.data = child.bias
            else:
                quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                            child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, q_type=q_type, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
            quant_conv_transpose.weight.data = child.weight
            module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            if child.bias is not None:
                quant_linear = QuantLinear(child.in_features, child.out_features, bias=True, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
                quant_linear.bias.data = child.bias
            else:
                quant_linear = QuantLinear(child.in_features, child.out_features, bias=False, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, quant_inference=quant_inference)
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear
        elif isinstance(child, nn.ReLU):
            quant_relu = QuantReLU(inplace=child.inplace, a_bits=a_bits, q_type=q_type, device=device)
            module._modules[name] = quant_relu
        elif isinstance(child, nn.Sigmoid):
            quant_sigmoid = QuantSigmoid(a_bits=a_bits, q_type=q_type, device=device)
            module._modules[name] = quant_sigmoid
        elif isinstance(child, nn.MaxPool2d):
            quant_max_pool = QuantMaxPool2d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding, a_bits=a_bits, q_type=q_type, device=device)
            module._modules[name] = quant_max_pool
        elif isinstance(child, nn.AvgPool2d):
            quant_avg_pool = QuantAvgPool2d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding, a_bits=a_bits, q_type=q_type, device=device)
            module._modules[name] = quant_avg_pool
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            quant_adaptive_avg_pool = QuantAdaptiveAvgPool2d(output_size=child.output_size, a_bits=a_bits, q_type=q_type, device=device)
            module._modules[name] = quant_adaptive_avg_pool
        else:
            add_quant_op(child, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, bn_fuse=bn_fuse, quant_inference=quant_inference)

def prepare(model, inplace=False, a_bits=8, w_bits=8, q_type=0, q_level=0, device='cpu', weight_observer=0, bn_fuse=0, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(model, a_bits=a_bits, w_bits=w_bits, q_type=q_type, q_level=q_level, device=device, weight_observer=weight_observer, bn_fuse=bn_fuse, quant_inference=quant_inference)
    return model

'''
# *** temp_dev ***
class QuantAdd(nn.Module):
    def __init__(self, a_bits=8, q_type=0):
        super(QuantAdd, self).__init__()
        if q_type == 0:
            self.activation_quantizer_0 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1, device=device)
            self.activation_quantizer_1 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1, device=device)
        else:
            self.activation_quantizer_0 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2, device=device)
            self.activation_quantizer_1 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2, device=device)

    def forward(self, shortcut, input):
        output = self.activation_quantizer_0(shortcut) + self.activation_quantizer_1(input)
        return output

class QuantConcat(nn.Module):
    def __init__(self, a_bits=8, q_type=0):
        super(QuantConcat, self).__init__()
        if q_type == 0:
            self.activation_quantizer_0 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1, device=device)
            self.activation_quantizer_1 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=1, device=device)
        else:
            self.activation_quantizer_0 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2, device=device)
            self.activation_quantizer_1 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L', out_channels=None, device=device), activation_weight_flag=2, device=device)

    def forward(self, shortcut, input):
        output = torch.cat((self.activation_quantizer_1(input), self.activation_quantizer_0(shortcut)), 1)
        return output
'''
