import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


# ********************* observers(统计min/max) *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':    # layer级(A/W)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # channel级(W)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == 'C_convtrans':  # W,min_max_shape=(1, C, 1, 1),channel级
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 0, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 0, keepdim=True)[0]
        elif self.q_level == 'FC':  # W,min_max_shape=(N, 1),channel级
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)

class MinMaxObserver(ObserverBase):  
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        if self.q_level == 'C':            # channel级
            self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        if self.q_level == 'C_convtrans':  
            self.register_buffer('min_val', torch.zeros(1, out_channels, 1, 1))
            self.register_buffer('max_val', torch.zeros(1, out_channels, 1, 1))
        if self.q_level == 'L':            # layer级
            self.register_buffer('min_val', torch.zeros(1))
            self.register_buffer('max_val', torch.zeros(1))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros(out_channels, 1))
            self.register_buffer('max_val', torch.zeros(out_channels, 1))
        self.num_flag = 0

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
    def __init__(self, q_level, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.num_flag = 0

    def update_range(self, min_val_cur, max_val_cur):
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
    def __init__(self, bits, observer):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.register_buffer('scale', torch.tensor(1.0))      # 量化比例因子
        self.register_buffer('zero_point', torch.tensor(0))   # 量化零点

    def update_params(self):
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
            self.observer(input)
            self.update_params()  # update scale and zero_point
            # 量化/反量化
            output = (torch.clamp(self.round(input / self.scale - self.zero_point), self.min_val, self.max_val) + self.zero_point) * self.scale
        return output

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1))

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))

# 对称量化
class SymmetricQuantizer(SignedQuantizer):
    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))                   # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))     # float_range
        self.scale = float_range / quantized_range                                                      # scale
        self.zero_point = torch.zeros_like(self.scale)                                                  # zero_point

# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_params(self):
        quantized_range = self.max_val - self.min_val                          # quantized_range
        float_range = self.observer.max_val - self.observer.min_val            # float_range
        self.scale = float_range / quantized_range                             # scale
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
                 first_layer=0):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        # 实例化量化器（A-layer级，W-channel级）
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels))
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels))
            else:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        quant_input = input
        quant_weight = self.weight_quantizer(self.weight) 
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
                 q_level=0):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, 
                         dilation, groups, bias, padding_mode)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C_convtrans', out_channels=out_channels))
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C_convtrans', out_channels=out_channels))
            else:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
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
                 first_layer=0):
        super(QuantBNFuseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                                bias, padding_mode)
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
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels))
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=out_channels))
            else:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_channels))
        self.first_layer = first_layer

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
          output = F.conv2d(quant_input, quant_weight, None, self.stride, self.padding, self.dilation,
                            self.groups) # 注意，这里不加bias（self.bias为None）
          # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
          output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
          output += reshape_to_activation(bias)
        else:  # 测试态
          output = F.conv2d(quant_input, quant_weight, bias, self.stride, self.padding, self.dilation,
                            self.groups) # 注意，这里加bias，做完整的conv+bn
        return output

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=8, w_bits=8, q_type=0, q_level=0):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features))
            else:
                self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_features))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            if q_level == 0:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='FC', out_channels=out_features))
            else:
                self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='L', out_channels=out_features))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        output = F.linear(quant_input, quant_weight, self.bias)
        return output
        
class QuantReLU(nn.ReLU):
    def __init__(self, inplace=False, a_bits=8, q_type=0):
        super(QuantReLU, self).__init__(inplace)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.relu(quant_input, self.inplace)
        return output

class QuantSigmoid(nn.Sigmoid):
    def __init__(self, a_bits=8, q_type=0):
        super(QuantSigmoid, self).__init__()
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.sigmoid(quant_input)
        return output

class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, a_bits=8, q_type=0):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.max_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)
        return output
        
class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, a_bits=8, q_type=0):
        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(quant_input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        return output
        
class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, a_bits=8, q_type=0):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.adaptive_avg_pool2d(quant_input, self.output_size)
        return output


'''
# *** temp_dev ***
class QuantAdd(nn.Module):
    def __init__(self, a_bits=8, q_type=0):
        super(QuantAdd, self).__init__()
        if q_type == 0:
            self.activation_quantizer_0 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            self.activation_quantizer_1 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer_0 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            self.activation_quantizer_1 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, shortcut, input):
        output = self.activation_quantizer_0(shortcut) + self.activation_quantizer_1(input)
        return output

class QuantConcat(nn.Module):
    def __init__(self, a_bits=8, q_type=0):
        super(QuantConcat, self).__init__()
        if q_type == 0:
            self.activation_quantizer_0 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            self.activation_quantizer_1 = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
        else:
            self.activation_quantizer_0 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))
            self.activation_quantizer_1 = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(q_level='L'))

    def forward(self, shortcut, input):
        output = torch.cat((self.activation_quantizer_1(input), self.activation_quantizer_0(shortcut)), 1)
        return output
        
class BNFold_Conv2d_Q(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        a_bits=8,
        w_bits=8,
        layer_index=0,
        weight_qpara_update=True,
        activation_qpara_update=True
    ):
        super().__init__()

        self.weight_qpara_update = weight_qpara_update
        self.activation_qpara_update = activation_qpara_update

        self.layer_index = layer_index

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(output_channels)

        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L',layer_index=self.layer_index), qpara_update = self.activation_qpara_update)
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=output_channels), qpara_update = self.weight_qpara_update)
    
    def forward(self, input):
        # 训练态
        if self.training:
            output = self.conv(input)
            output = self.bn(output)
            if self.conv.bias is not None:  
              bias = reshape_to_bias(self.bn.bias + (self.conv.bias -  self.bn.running_mean) * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)))
            else:
              bias = reshape_to_bias(self.bn.bias - self.bn.running_mean  * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)))
            weight = self.conv.weight * reshape_to_weight(self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps))     
        # 测试态
        else:
            if self.conv.bias is not None:  
              bias = reshape_to_bias(self.bn.bias + (self.conv.bias -  self.bn.running_mean) * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)))
            else:
              bias = reshape_to_bias(self.bn.bias - self.bn.running_mean  * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)))
            weight = self.conv.weight * reshape_to_weight(self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps))     
            # +++
            # 所有weight处理
            weight += 1e-9
            # 极小weight处理
            #mask_le = torch.abs(weight).le(1e-9)
            #weight[mask_le] +=  1e-9

        q_input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(weight)

        output = F.conv2d(
              input=q_input,
              weight=q_weight,
              bias=bias,  
              stride=self.conv.stride,
              padding=self.conv.padding,
              dilation=self.conv.dilation,
              groups=self.conv.groups
          )
        return output

class BNFold_ConvTrans2d_Q_ReLU(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=False,
        dilation=1,
        a_bits=8,
        w_bits=8,
        layer_index=0,
        weight_qpara_update=True,
        activation_qpara_update=True
    ):
        super().__init__()

        self.weight_qpara_update = weight_qpara_update
        self.activation_qpara_update = activation_qpara_update

        self.layer_index = layer_index
        
        #self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.up_conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        self.up_bn = nn.BatchNorm2d(output_channels)
        self.up_relu = nn.ReLU(inplace=True)

        # 实例化量化器（A-layer级，W-channel级）
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=HistogramObserver(q_level='L',layer_index=self.layer_index), qpara_update = self.activation_qpara_update)
        #self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C', out_channels=output_channels), qpara_update = self.weight_qpara_update)
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(q_level='C_convtrans', out_channels=output_channels), qpara_update = self.weight_qpara_update)

    def forward(self, input):
        # 训练态
        if self.training:
            output = self.up_conv(input)
            output = self.up_bn(output)
            if self.up_conv.bias is not None:  
              bias = reshape_to_bias(self.up_bn.bias + (self.up_conv.bias -  self.up_bn.running_mean) * (self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps)))
            else:
              bias = reshape_to_bias(self.up_bn.bias - self.up_bn.running_mean  * (self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps)))# b融batch
            #weight = self.up_conv.weight * reshape_to_weight(self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps))     # w融running
            weight = self.up_conv.weight * reshape_to_convtrans_weight(self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps))     # w融running
        # 测试态
        else:
            if self.up_conv.bias is not None:  
              bias = reshape_to_bias(self.up_bn.bias + (self.up_conv.bias -  self.up_bn.running_mean) * (self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps)))
            else:
              bias = reshape_to_bias(self.up_bn.bias - self.up_bn.running_mean  * (self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps)))# b融batch
            #weight = self.up_conv.weight * reshape_to_weight(self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps))     # w融running
            weight = self.up_conv.weight * reshape_to_convtrans_weight(self.up_bn.weight / torch.sqrt(self.up_bn.running_var + self.up_bn.eps))     # w融running

        q_input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(weight)

        output = F.conv_transpose2d(
                q_input, q_weight, bias, self.up_conv.stride, self.up_conv.padding,
                self.up_conv.output_padding, self.up_conv.groups, self.up_conv.dilation)

        output = self.up_relu(output) 
        return output
'''
