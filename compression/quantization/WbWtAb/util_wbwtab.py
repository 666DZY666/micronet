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
class ActivationBin(nn.Module):
    def __init__(self, A):
        super(ActivationBin, self).__init__()
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
class WeightTnnBin(nn.Module):
    def __init__(self, W):
        super(WeightTnnBin, self).__init__()
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
                 A=2,
                 W=2):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        # 实例化调用A和W量化器
        self.activation_quantizer = ActivationBin(A=A)
        self.weight_quantizer = WeightTnnBin(W=W)
          
    def forward(self, input):
        # 量化A和W
        bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)    
        # 用量化后的A和W做卷积
        output = F.conv2d(bin_input, tnn_bin_weight, self.bias, self.stride, self.padding, self.dilation, 
                          self.groups)
        return output
        