import torch
import torch.nn as nn
import time
import sys
import numpy as np
import nin_gc
from layers import bn
import argparse

# ******************** 是否保存模型完整参数 ********************
#torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse(conv, bn):
    global i
    i = i + 1
    # ******************** BN参数 *********************
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias
    # ******************* conv参数 ********************
    w = conv.weight
    w_fold = w.clone()
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    b_fold = b.clone()

    # ******************* 针对特征(A)二值的BN融合 *******************
    if(i <= 7):
        mask_positive = gamma.data.gt(0)
        mask_negetive = gamma.data.lt(0)

        w_fold[mask_positive] = w[mask_positive]
        b_fold[mask_positive] = b[mask_positive] - mean[mask_positive] + beta[mask_positive] * (var_sqrt[mask_positive] / gamma[mask_positive])

        w_fold[mask_negetive] = w[mask_negetive] * -1
        b_fold[mask_negetive] = mean[mask_negetive] - b[mask_negetive] - beta[mask_negetive] * (var_sqrt[mask_negetive] / gamma[mask_negetive])
    # ******************* 普通BN融合 *******************
    else:
        w_fold = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b_fold = (b - mean) * (gamma / var_sqrt) + beta
    
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups=conv.groups,
                         bias=True)
    fused_conv.weight = nn.Parameter(w_fold)
    fused_conv.bias = nn.Parameter(b_fold)
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

# ********************** 可选配置参数 **********************
parser = argparse.ArgumentParser()
# W —— 三值/二值(据训练时W量化(三/二值)情况而定)
parser.add_argument('--W', type=int, default=2,
            help='Wb:2, Wt:3')
args = parser.parse_args()
print('==> Options:',args)

i = 0
print("************* 参数量化 + BN_fuse **************")
# ********************** W全精度表示 ************************
model = nin_gc.Net()
#print(model)
model.load_state_dict(torch.load("../models_save/nin_gc_bn_gama.pth")['state_dict'])
torch.save(model, 'models_save/model.pth')
torch.save(model.state_dict(), 'models_save/model_para.pth')
model_array = np.array(model)
model_para_array = np.array(model.state_dict())
np.savetxt('models_save/model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

# ********************** W量化表示(据训练时W量化(三/二值)情况而定) *************************
if args.W == 2 or args.W == 3:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            i = i + 1
            if (i >= 2 and i <= 8):
                # **************** channel级 - E(|W|) ****************
                E = torch.mean(torch.abs(m.weight.data), (3, 2, 1), keepdim=True)
                # **************************************** W二值 *****************************************
                if args.W == 2:
                    # **************** α(缩放因子) ****************
                    alpha = E
                    # ************** W —— +-1 **************
                    m.weight.data = m.weight.data.sign()
                    # ************** W * α **************
                    m.weight.data = m.weight.data * alpha
                # **************************************** W三值 *****************************************
                elif args.W == 3:
                    # **************** 阈值 ****************
                    threshold = E * 0.7
                    # **************** α(缩放因子) ****************
                    a_abs = m.weight.data.abs().clone()
                    mask_le = a_abs.le(threshold)
                    mask_gt = a_abs.gt(threshold)
                    a_abs[mask_le] = 0
                    a_abs_th = a_abs.clone()
                    a_abs_th_sum = torch.sum(a_abs_th, (3, 2, 1), keepdim=True)
                    mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                    alpha = a_abs_th_sum / mask_gt_sum
                    # ************** W —— +-1、0 **************
                    m.weight.data = torch.sign(torch.add(torch.sign(torch.add(m.weight.data, threshold)),torch.sign(torch.add(m.weight.data, -threshold))))
                    # *************** W * α ************************
                    m.weight.data = m.weight.data * alpha

i = 0
torch.save(model, 'models_save/quan_model.pth')
torch.save(model.state_dict(), 'models_save/quan_model_para.pth')
model_array = np.array(model)
model_para_array = np.array(model.state_dict())
np.savetxt('models_save/quan_model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/quan_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

# ********************* BN融合 **********************
model.eval()
model_fused = fuse_model(model)
torch.save(model_fused, 'models_save/quan_bn_merge_model.pth')
torch.save(model_fused.state_dict(), 'models_save/quan_bn_merge_model_para.pth')
model_array = np.array(model_fused)
model_para_array = np.array(model_fused.state_dict())
np.savetxt('models_save/quan_bn_merge_model.txt', [model_array], fmt = '%s', delimiter=',')
np.savetxt('models_save/quan_bn_merge_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')

# *********************** 转换测试 *************************
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
