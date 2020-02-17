import torch.nn as nn
import numpy
import torch

# 权重（W）量化（三值或二值）
class Tnn_Bin_Op():
    def __init__(self, model, W=2):
        self.W = W
        # 统计conv数量
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1
        # 仅量化中间层filter(首尾层W|A均不量化)
        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
    # 权重（参数）量化（三值或二值）
    def tnn_bin(self):
        self.meancenterConvParams() # W中心化
        self.clampConvParams()      # W截断
        self.save_params()          # 保存浮点W
        self.tnn_bin_ConvParams()   # W量化

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    # ************************** W量化（三值或二值）******************************
    def tnn_bin_ConvParams(self):
        if self.W == 2 or self.W == 3:
            for index in range(self.num_of_params):
                # **************** channel级 - E(|W|) ****************
                E = torch.mean(torch.abs(self.target_modules[index].data), (3, 2, 1), keepdim=True)
                # **************************************** W二值 *****************************************
                if self.W == 2:                
                    # **************** α(缩放因子) ****************
                    alpha = E
                    # ************** W —— +-1 **************
                    self.target_modules[index].data = self.target_modules[index].data.sign()
                    # ************** W * α **************
                    self.target_modules[index].data = self.target_modules[index].data * alpha
                elif self.W == 3:
                    # **************************************** W三值 *****************************************
                    # **************** 阈值 ****************
                    threshold = E * 0.7
                    # **************** α(缩放因子) ****************
                    a_abs = self.target_modules[index].data.abs().clone()
                    mask_le = a_abs.le(threshold)
                    mask_gt = a_abs.gt(threshold)
                    a_abs[mask_le] = 0
                    a_abs_th = a_abs.clone()
                    a_abs_th_sum = torch.sum(a_abs_th, (3, 2, 1), keepdim=True)
                    mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                    alpha = a_abs_th_sum / mask_gt_sum
                    # ************** W —— +-1、0 **************
                    self.target_modules[index].data = torch.sign(torch.add(torch.sign(torch.add(self.target_modules[index].data, threshold)),torch.sign(torch.add(self.target_modules[index].data, -threshold))))
                    # *************** W * α ************************
                    self.target_modules[index].data = self.target_modules[index].data * alpha
                #print(self.target_modules[index].data)

    # 恢复浮点W
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    
    # α（缩放因子） ——> grad
    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
