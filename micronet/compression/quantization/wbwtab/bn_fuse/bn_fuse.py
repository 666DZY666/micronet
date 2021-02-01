import copy
import sys
sys.path.append("..")
sys.path.append("../../../..")
import numpy as np
import argparse
import torch
import torch.nn as nn
from models import nin_gc, nin

import quantize


# ******************** 是否保存模型完整参数 ********************
#torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)

# BN融合
def bn_fuse(conv, bn):
    # 可以进行“针对特征(A)二值的BN融合”的BN层位置
    global bn_counter, bin_bn_fuse_num
    bn_counter = bn_counter + 1
    # ******************** BN参数 *********************
    mean = bn.running_mean
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias
    # ******************* conv参数 ********************
    w = conv.weight
    w_fused = w.clone()
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    b_fused = b.clone()
    # ******************* 针对特征(A)二值的BN融合 *******************
    if(bn_counter >= 1 and bn_counter <= bin_bn_fuse_num):
        mask_positive = gamma.data.gt(0)
        mask_negetive = gamma.data.lt(0)

        w_fused[mask_positive] = w[mask_positive]
        b_fused[mask_positive] = b[mask_positive] - mean[mask_positive] + beta[mask_positive] * (std[mask_positive] / gamma[mask_positive])

        w_fused[mask_negetive] = w[mask_negetive] * -1
        b_fused[mask_negetive] = mean[mask_negetive] - b[mask_negetive] - beta[mask_negetive] * (std[mask_negetive] / gamma[mask_negetive])
    # ******************* 普通BN融合 *******************
    else:
        w_fused = w * (gamma / std).reshape([conv.out_channels, 1, 1, 1])
        b_fused = beta + (b - mean) * (gamma / std) 
    # 新建bn_fused_conv(BN融合的Conv层),将w_fused,b_fused赋值于其参数
    bn_fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups=conv.groups,
                         bias=True)
    bn_fused_conv.weight.data = w_fused
    bn_fused_conv.bias.data = b_fused
    return bn_fused_conv

# 模型BN融合
def bn_fuse_module(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            conv_name_temp = name
            conv_child_temp = child
        elif isinstance(child, nn.BatchNorm2d):
            bn_fused_conv = bn_fuse(conv_child_temp, child) # BN融合
            module._modules[conv_name_temp] = bn_fused_conv
            module._modules[name] = nn.Identity()
        else:
            bn_fuse_module(child)

def model_bn_fuse(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    bn_fuse_module(model)
    return model

if __name__=='__main__':
    # ********************** 可选配置参数 **********************
    parser = argparse.ArgumentParser()
    # cpu、gpu
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    # prune_quant
    parser.add_argument('--prune_quant', action='store_true',
            help='this is prune_quant model')
    args = parser.parse_args()
    print('==> Options:',args)

    # ********************** 模型加载 ************************
    if args.prune_quant:
        print('******Prune Quant model******')
        ori_model = nin_gc.Net(cfg=torch.load('../models_save/nin_gc.pth')['cfg'])
    else:
        ori_model = nin_gc.Net()
    if not args.cpu:
        ori_model.load_state_dict(torch.load('../models_save/nin_gc.pth')['state_dict'])
    else:
        ori_model.load_state_dict(torch.load('../models_save/nin_gc.pth', map_location='cpu')['state_dict'])
    quant_model = quantize.prepare(ori_model, inplace=False, A=2)
    
    # ********************** ori_model ************************
    torch.save(ori_model, 'models_save/model.pth')
    torch.save(ori_model.state_dict(), 'models_save/model_para.pth')
    model_array = np.array(ori_model)
    model_para_array = np.array(ori_model.state_dict())
    np.savetxt('models_save/model.txt', [model_array], fmt = '%s', delimiter=',')
    np.savetxt('models_save/model_para.txt', [model_para_array], fmt = '%s', delimiter=',')
    
    # ********************* bn_fused_model **********************
    bn_counter = 0
    bin_bn_fuse_num = 0
    for m in quant_model.modules():
        if isinstance(m, quantize.ActivationQuantizer):
            bin_bn_fuse_num += 1                             # 统计可以进行“针对特征(A)二值的BN融合”的BN层位置
    bn_fused_model = model_bn_fuse(ori_model, inplace=False) #  模型BN融合
    print('***ori_model***\n', ori_model)
    print('\n***bn_fused_model***\n', bn_fused_model)
    torch.save(bn_fused_model, 'models_save/bn_fused_model.pth')                   # 保存量化融合模型(结构+参数)
    torch.save(bn_fused_model.state_dict(), 'models_save/bn_fused_model_para.pth') # 保存量化融合模型参数
    model_array = np.array(bn_fused_model)
    model_para_array = np.array(bn_fused_model.state_dict())
    np.savetxt('models_save/bn_fused_model.txt', [model_array], fmt = '%s', delimiter=',')
    np.savetxt('models_save/bn_fused_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')
    print("************* bn_fuse 完成 **************")
    print("************* bn_fused_model 已保存 **************")
