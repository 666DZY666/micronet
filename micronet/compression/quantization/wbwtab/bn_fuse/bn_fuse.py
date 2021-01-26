import sys
sys.path.append("..")
import numpy as np
import argparse
import torch
import torch.nn as nn
import nin_gc_inference
import nin_gc_training

import quantize

# ******************** 是否保存模型完整参数 ********************
#torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)

# 原BN替代层
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# BN融合
def bn_fuse(conv, bn):
    # 可以进行“针对特征(A)二值的BN融合”的BN层位置
    global bn_counter, bn_fuse_num
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
    if(bn_counter >= 1 and bn_counter <= bn_fuse_num):
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
def model_bn_fuse(model):
    children = list(model.named_children())
    name_temp = None
    child_temp = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bn_fused_conv = bn_fuse(child_temp, child) # BN融合
            model._modules[name_temp] = bn_fused_conv
            model._modules[name] = Identity()
            child_temp = None
        elif isinstance(child, nn.Conv2d):
            name_temp = name
            child_temp = child
        else:
            model_bn_fuse(child)
    return model

if __name__=='__main__':
    # ********************** 可选配置参数 **********************
    parser = argparse.ArgumentParser()
    # cpu、gpu
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    # W —— 三值/二值(据训练时W量化(三/二值)情况而定)
    parser.add_argument('--W', type=int, default=2,
            help='Wb:2, Wt:3')
    args = parser.parse_args()
    print('==> Options:',args)

    weight_quantizer = quantize.WeightQuantizer(W=args.W)  # 实例化W量化器

    # ********************** 模型加载 ************************
    model_0 = nin_gc_training.Net()
    quantize.prepare(model_0, inplace=True)
    model_1 = nin_gc_inference.Net()
    if not args.cpu:
        model_0.load_state_dict(torch.load('../models_save/nin_gc.pth')['state_dict'])
    else:
        model_0.load_state_dict(torch.load('../models_save/nin_gc.pth', map_location='cpu')['state_dict'])
    # ********************** W全精度表示 ************************
    torch.save(model_0, 'models_save/model.pth')
    torch.save(model_0.state_dict(), 'models_save/model_para.pth')
    model_array = np.array(model_0)
    model_para_array = np.array(model_0.state_dict())
    np.savetxt('models_save/model.txt', [model_array], fmt = '%s', delimiter=',')
    np.savetxt('models_save/model_para.txt', [model_para_array], fmt = '%s', delimiter=',')
    # ********************** W量化表示(据训练时W量化(三/二值)情况而定) *************************
    bn_fuse_num = 0
    for m in model_0.modules():
        if isinstance(m, quantize.ActivationQuantizer):
            bn_fuse_num += 1                            # 统计可以进行“针对特征(A)二值的BN融合”的BN层位置
        if isinstance(m, quantize.QuantConv2d):
            m.weight.data = weight_quantizer(m.weight)  # W量化表示
    torch.save(model_0.state_dict(), 'models_save/quant_model_para.pth')  # 保存量化模型参数
    model_array = np.array(model_0)
    model_para_array = np.array(model_0.state_dict())
    np.savetxt('models_save/quant_model.txt', [model_array], fmt = '%s', delimiter=',')
    np.savetxt('models_save/quant_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')
    print("************* 参数量化表示 - 完成 **************")

    # ********************* BN融合 **********************
    bn_counter = 0
    model_1.load_state_dict(torch.load('models_save/quant_model_para.pth'))
    torch.save(model_1, 'models_save/quant_model.pth')                    # 保存量化模型(结构+参数)
    quant_bn_fused_model = model_bn_fuse(model_1) #  模型BN融合
    torch.save(quant_bn_fused_model, 'models_save/quant_bn_fused_model.pth')                   # 保存量化融合模型(结构+参数)
    torch.save(quant_bn_fused_model.state_dict(), 'models_save/quant_bn_fused_model_para.pth') # 保存量化融合模型参数
    model_array = np.array(quant_bn_fused_model)
    model_para_array = np.array(quant_bn_fused_model.state_dict())
    np.savetxt('models_save/quant_bn_fused_model.txt', [model_array], fmt = '%s', delimiter=',')
    np.savetxt('models_save/quant_bn_fused_model_para.txt', [model_para_array], fmt = '%s', delimiter=',')
    print("************* bn_fuse - 完成 **************")

    # *********************** 转换预测试(dataset测试在bn_fused_model_test.py中进行) *************************
    quant_model = nin_gc_inference.Net()
    quant_model.load_state_dict(torch.load('models_save/quant_model_para.pth'))    # 加载量化模型
    quant_model.eval()
    quant_bn_fused_model.eval()
    softmax = nn.Softmax(dim=1)
    f = 0
    epochs = 100
    print("\r\n************* 转换预测试 **************")
    for i in range(0, epochs):
        p = torch.rand([1, 3, 32, 32])
        out = softmax(quant_model(p))                       # 量化模型测试
        out_bn_fused = softmax(quant_bn_fused_model(p)) # 量化融合模型测试
        #print(out_bn_fused)
        if(out.argmax() == out_bn_fused.argmax()):
            f += 1
    print('The last result:')
    print('quant_model_output:', out)
    print('quant_bn_fused_model_output:', out_bn_fused)
    print("bn_fuse_success_rate: {:.2f}%".format((f / epochs) * 100))
