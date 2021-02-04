import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ********************* 二值(+-1) ***********************
# activation
class BinaryActivation(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        # ******************** A —— 1、0 *********************
        #output = torch.clamp(output, min=0)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        #*******************ste*********************
        grad_input = grad_output.clone()
        #****************saturate_ste***************
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        '''
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        '''
        return grad_input

# weight
class BinaryWeight(Function):
    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        #*******************ste*********************
        grad_input = grad_output.clone()
        return grad_input

# ********************* 三值(+-1、0) ***********************
class Ternary(Function):
    @staticmethod
    def forward(self, input):
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(torch.add(torch.sign(torch.add(input, threshold)),torch.sign(torch.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        #*******************ste*********************
        grad_input = grad_output.clone()
        return grad_input

# ********************* A(特征)量化(二值) ***********************
class ActivationQuantizer(nn.Module):
    def __init__(self, A=2):
        super(ActivationQuantizer, self).__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True) 
    def binary(self, input):
        output = BinaryActivation.apply(input)
        return output 
    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
        else:
            output = self.relu(input)
        return output

# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clamp_convparams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub_(mean)        # W中心化(C方向)
    w.data.clamp_(-1.0, 1.0) # W截断
    return w
class WeightQuantizer(nn.Module):
    def __init__(self, W=2):
        super(WeightQuantizer, self).__init__()
        self.W = W    
    def binary(self, input):
        output = BinaryWeight.apply(input)
        return output 
    def ternary(self, input):
        output = Ternary.apply(input)
        return output 
    def forward(self, input):
        if self.W == 2 or self.W == 3:
            # **************************************** W二值 *****************************************
            if self.W == 2:
                output = meancenter_clamp_convparams(input) # W中心化+截断
                # **************** channel级 - E(|W|) ****************
                E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
                # **************** α(缩放因子) ****************
                alpha = E
                # ************** W —— +-1 **************
                output = self.binary(output)
                # ************** W * α **************
                output = output * alpha # 若不需要α(缩放因子)，注释掉即可
                # **************************************** W三值 *****************************************
            elif self.W == 3:
                output_fp = input.clone()
                # ************** W —— +-1、0 **************
                output, threshold = self.ternary(input) # threshold(阈值)
                # **************** α(缩放因子) ****************
                output_abs = torch.abs(output_fp)
                mask_le = output_abs.le(threshold)
                mask_gt = output_abs.gt(threshold)
                output_abs[mask_le] = 0
                output_abs_th = output_abs.clone()
                output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
                mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                alpha = output_abs_th_sum / mask_gt_sum # α(缩放因子)
                # *************** W * α ****************
                output = output * alpha # 若不需要α(缩放因子)，注释掉即可
        else:
            output = input
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
                 W=2,
                 quant_inference=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)
        
    def forward(self, input):
        if not self.quant_inference:
            tnn_bin_weight = self.weight_quantizer(self.weight) 
        else:
            tnn_bin_weight = self.weight
        output = F.conv2d(input, tnn_bin_weight, self.bias, self.stride, self.padding, self.dilation, 
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
                 W=2,
                 quant_inference=False):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, 
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)

    def forward(self, input):
        if not self.quant_inference:
            tnn_bin_weight = self.weight_quantizer(self.weight) 
        else:
            tnn_bin_weight = self.weight
        output = F.conv_transpose2d(input, tnn_bin_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output

def add_quant_op(module, layer_counter, layer_num, A=2, W=2, quant_inference=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1 and layer_counter[0] < layer_num:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, W=W, quant_inference=quant_inference)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, W=W, quant_inference=quant_inference)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1 and layer_counter[0] < layer_num:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=True, padding_mode=child.padding_mode, W=W, quant_inference=quant_inference)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride, padding=child.padding, output_padding=child.output_padding, dilation=child.dilation, groups=child.groups, bias=False, padding_mode=child.padding_mode, W=W, quant_inference=quant_inference)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.ReLU):
            if layer_counter[0] > 0 and layer_counter[0] < layer_num:
                quant_relu = ActivationQuantizer(A=A)
                module._modules[name] = quant_relu
        else:
            add_quant_op(child, layer_counter, layer_num, A=A, W=W, quant_inference=quant_inference)

def prepare(model, inplace=False, A=2, W=2, quant_inference=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    layer_num = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layer_num += 1
        elif isinstance(m, nn.ConvTranspose2d):
            layer_num += 1
    add_quant_op(model, layer_counter, layer_num, A=A, W=W, quant_inference=quant_inference)
    return model
        