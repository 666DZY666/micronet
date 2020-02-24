import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ********************* 二值(+-1) ***********************
class Binary(Function):

    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

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

    def forward(self, input):
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(torch.add(torch.sign(torch.add(input, threshold)),torch.sign(torch.add(input, -threshold))))
        return output, threshold

    def backward(self, grad_output, grad_threshold):
        #*******************ste*********************
        grad_input = grad_output.clone()
        return grad_input

# ********************* A(特征)量化(二值) ***********************
class activation_bin(nn.Module):
  def __init__(self, A):
    super(activation_bin, self).__init__()
    self.A = A
    self.relu = nn.ReLU(inplace=True)

  def forward(self, a):
    if self.A == 2:
      a = Binary()(a)
      # ******************** A —— 1、0 *********************
      #a = torch.clamp(a, min=0)
    else:
      a = self.relu(a)
    return a
# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clampConvParams(w):
    mean = torch.mean(w, 1, keepdim=True)
    w = torch.sub(w, mean)# W中心化
    w = torch.clamp(w, min=-1.0, max=1.0)# W截断
    return w
class weight_tnn_bin(nn.Module):
  def __init__(self, W):
    super(weight_tnn_bin, self).__init__()
    self.W = W

  def forward(self, w):
    if self.W == 2 or self.W == 3:
        # **************************************** W二值 *****************************************
        if self.W == 2:
            w = meancenter_clampConvParams(w)# W中心化+截断
            # **************** channel级 - E(|W|) ****************
            E = torch.mean(torch.abs(w), (3, 2, 1), keepdim=True)
            # **************** α(缩放因子) ****************
            alpha = E
            # ************** W —— +-1 **************
            w = Binary()(w)
            # ************** W * α **************
            w = w * alpha # 若不需要α(缩放因子)，注释掉即可
            # **************************************** W三值 *****************************************
        elif self.W == 3:
            w_fp = w.clone()
            # ************** W —— +-1、0 **************
            w, threshold = Ternary()(w)
            # **************** α(缩放因子) ****************
            a_abs = torch.abs(w_fp)
            mask_le = a_abs.le(threshold)
            mask_gt = a_abs.gt(threshold)
            a_abs[mask_le] = 0
            a_abs_th = a_abs.clone()
            a_abs_th_sum = torch.sum(a_abs_th, (3, 2, 1), keepdim=True)
            mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
            alpha = a_abs_th_sum / mask_gt_sum # α(缩放因子)
            # *************** W * α ****************
            w = w * alpha # 若不需要α(缩放因子)，注释掉即可
    return w

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
        super(Conv2d_Q, self).__init__(
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
