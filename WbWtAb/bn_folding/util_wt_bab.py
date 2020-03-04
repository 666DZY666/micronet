import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ********************* 二值(+-1) ***********************
class Binary(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
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
class activation_bin(nn.Module):
  def __init__(self, A):
    super().__init__()
    self.A = A
    self.relu = nn.ReLU(inplace=True)

  def binary(self, input):
    output = Binary.apply(input)
    return output

  def forward(self, input):
    if self.A == 2:
      output = self.binary(input)
      # ******************** A —— 1、0 *********************
      #a = torch.clamp(a, min=0)
    else:
      output = self.relu(input)
    return output
# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clampConvParams(w):
    mean = torch.mean(w, 1, keepdim=True)
    w = torch.sub(w, mean)# W中心化
    w = torch.clamp(w, min=-1.0, max=1.0)# W截断
    return w
class weight_tnn_bin(nn.Module):
  def __init__(self, W):
    super().__init__()
    self.W = W

  def binary(self, input):
    output = Binary.apply(input)
    return output

  def ternary(self, input):
    output = Ternary.apply(input)
    return output

  def forward(self, input):
    if self.W == 2 or self.W == 3:
        # **************************************** W二值 *****************************************
        if self.W == 2:
            output = meancenter_clampConvParams(input)# W中心化+截断
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
            output, threshold = self.ternary(input)
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
class Conv2d_Q(nn.Conv2d):
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
        A=2,
        W=2
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
        self.activation_quantizer = activation_bin(A=A)
        self.weight_quantizer = weight_tnn_bin(W=W)
          
    def forward(self, input):
        # 量化A和W
        bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)    
        #print(bin_input)
        #print(tnn_bin_weight)
        # 用量化后的A和W做卷积
        output = F.conv2d(
            input=bin_input, 
            weight=tnn_bin_weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups)
        return output
