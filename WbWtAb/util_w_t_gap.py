import torch.nn as nn
import numpy
import torch

class Tnn_Bin_Op():
    def __init__(self, model):
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1
        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.saved_params_0 = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    tmp_0 = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
                    n = self.target_modules[index-1].data[0].nelement()
                    s = self.target_modules[index-1].data.size()
                    m = self.target_modules[index-1].data.norm(1, 3, keepdim=True)\
                            .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                    #********************** W三值 *************************
                    for i in range(0, s[0]):
                        sum = torch.sum(torch.abs(self.target_modules[index-1].data[i])).item()
                        threshold = (sum / n) * 0.7
                        #threshold = 0.7 * self.target_modules[index-1].data[i].norm(1).div(n).item()
                        #threshold = 0.7 * torch.mean(torch.abs(self.target_modules[index-1].data[i])).item()
                        tmp_0[i] = torch.sign(torch.add(torch.sign(torch.add(self.target_modules[index-1].data[i], threshold)),torch.sign(torch.add(self.target_modules[index-1].data[i], -threshold))))
                    self.saved_params_0.append(tmp_0)             
        #print(self.saved_params_0)

    # 权重（参数）量化（二值或三值）
    def tnn_bin(self):
        self.meancenterConvParams() # W中心化
        self.clampConvParams()      # W截断
        self.save_params()          # 保存浮点W
        self.tnn_bin_ConvParams()   # W量化
        self.save_params_0()        

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
    
    def save_params_0(self):
        for index in range(self.num_of_params):
            self.saved_params_0[index].copy_(self.target_modules[index].data)

    # *************************W量化（三值_include gap）******************************
    def tnn_bin_ConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            for i in range(0, s[0]):
                sum = torch.sum(torch.abs(self.target_modules[index].data[i])).item()
                threshold = (sum / n) * 0.7
                #threshold = 0.7 * self.target_modules[index].data[i].norm(1).div(n).item()
                #threshold = 0.7 * torch.mean(torch.abs(self.target_modules[index].data[i])).item()
                gap_threshold = (4 / 5) * threshold #gap:4/5、3/4、2/3
                #print(threshold, gap_threshold)

                idx_0 = self.target_modules[index].data[i].ge(threshold)
                idx_1 = self.target_modules[index].data[i].le(-threshold)
                idx_3 = self.target_modules[index].data[i].gt(gap_threshold)
                idx_4 = self.target_modules[index].data[i].lt(-gap_threshold)
                idx_5 = 1 - (idx_3 + idx_4)
                idx_6 = idx_3 - idx_0
                idx_7 = idx_4 - idx_1
                idx_8 = idx_6 + idx_7
                self.target_modules[index].data[i][idx_0] = 1
                self.target_modules[index].data[i][idx_1] = -1
                self.target_modules[index].data[i][idx_5] = 0
                self.target_modules[index].data[i][idx_8] = self.saved_params_0[index][i][idx_8]

    # 恢复浮点W
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
