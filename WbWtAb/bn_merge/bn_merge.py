import torch
import torch.nn as nn
import time
import sys
import numpy as np
import nin_gc
from layers import bn
import argparse

#torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse(conv, bn):
    global i
    i = i + 1
    # *******************conv参数********************
    w = conv.weight
    b = conv.bias
    # ********************BN参数*********************
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias

    if(i >= 2 and i <= 7):
        b = b - mean + beta * var_sqrt
    else:
        w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean)/var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups=conv.groups,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None
    for name, child in children:
        #if isinstance(child, nn.BatchNorm2d):
        if isinstance(child, bn.BatchNorm2d_bin):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)

def fuse_model(m):
    p = torch.rand([1, 3, 32, 32])
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)
    return m

#**********************参数定义**********************
parser = argparse.ArgumentParser()
# W —— 三值/二值(据训练时W量化(三/二值)情况而定)
parser.add_argument('--W', type=int, default=2,
            help='Wb:2, Wt:3')
args = parser.parse_args()
print('==> Options:',args)

i = 0
print("************* 参数量化 + BN_fuse **************")
#**********************W全精度表示************************
model = nin_gc.Net()
#print(model)
model.load_state_dict(torch.load("../models_save/nin_gc_bn_gama.pth")['state_dict'])
torch.save(model, 'models_save/model.pth')
torch.save(model.state_dict(), 'models_save/model_para.pth')
model_array = np.array(model)
model_para_array = np.array(model.state_dict())
np.savetxt('models_save/model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

#********************** W量化表示(据训练时W量化(三/二值)情况而定)*************************
if args.W == 2 or args.W == 3:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            i = i + 1
            if (i >= 2 and i <= 8):
                n = m.weight.data[0].nelement()
                s = m.weight.data.size()
                if args.W == 2:
                    #****************************************W二值*****************************************
                    # ****************α****************
                    alpha = m.weight.data.norm(1, 3, keepdim=True)\
                            .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                    # **************** W —— +-1 ****************
                    m.weight.data = m.weight.data.sign()
                    # **************** W * α ****************
                    m.weight.data = m.weight.data * alpha
                elif args.W == 3:
                    #****************************************W三值****************************************
                    for j in range(0, s[0]):
                            sum = torch.sum(torch.abs(m.weight.data[j])).item()
                            threshold = (sum / n) * 0.7
                            #threshold = 0.7 * self.target_modules[index].data[i].norm(1).div(n).item()
                            #threshold = 0.7 * torch.mean(torch.abs(self.target_modules[index].data[i])).item()
                            # ****************α****************
                            a_abs = m.weight.data[j].abs().clone()
                            mask = a_abs.gt(threshold)
                            a_abs_th = a_abs[mask].clone()
                            alpha = torch.mean(a_abs_th)
                            #print(threshold, alpha)
                            # **************** W —— +-1、0 ****************
                            m.weight.data[j] = torch.sign(torch.add(torch.sign(torch.add(m.weight.data[j], threshold)),torch.sign(torch.add(m.weight.data[j], -threshold))))
                            # **************** W * α ****************
                            m.weight.data[j] = m.weight.data[j] * alpha
i = 0
torch.save(model, 'models_save/quan_model.pth')
torch.save(model.state_dict(), 'models_save/quan_model_para.pth')
model_array = np.array(model)
model_para_array = np.array(model.state_dict())
np.savetxt('models_save/quan_model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/quan_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

#********************* BN融合 **********************
model.eval()
model_fused = fuse_model(model)
torch.save(model_fused, 'models_save/quan_bn_merge_model.pth')
torch.save(model_fused.state_dict(), 'models_save/quan_bn_merge_model_para.pth')
model_array = np.array(model_fused)
model_para_array = np.array(model_fused.state_dict())
np.savetxt('models_save/quan_bn_merge_model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/quan_bn_merge_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

#***********************转换测试*************************
model = torch.load('models_save/quan_model.pth')
model.eval()
model_fused = torch.load('models_save/quan_bn_merge_model.pth')
model_fused.eval()
m = nn.Softmax()
f = 0
epochs = 100
for i in range(0, epochs):
    p = torch.rand([1, 3, 32, 32])
    out = m(model(p))
    out_fused = m(model_fused(p))

    if(out.argmax() == out_fused.argmax()):
        f += 1

print('The last result:\r\n')
print(out)
print(out_fused, '\r\n')
print("merge_success_rate: {:.2f}%".format((f / epochs) * 100))