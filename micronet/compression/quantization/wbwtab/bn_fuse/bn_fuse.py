import copy
import sys

sys.path.append("..")
sys.path.append("../../../..")
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import nin_gc, nin

import quantize


# ******************** 是否保存模型完整参数 ********************
# torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)


def bn_fuse(conv, bn):
    # 可以进行“针对特征(A)二值的BN融合”的BN层位置
    global bn_counter, bin_bn_fuse_num
    bn_counter = bn_counter + 1
    # ******************** bn参数 *********************
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
    # ******************* 针对特征(A)二值的bn融合 *******************
    if bn_counter >= 1 and bn_counter <= bin_bn_fuse_num:
        mask_positive = gamma.data.gt(0)
        mask_negetive = gamma.data.lt(0)

        w_fused[mask_positive] = w[mask_positive]
        b_fused[mask_positive] = (
            b[mask_positive]
            - mean[mask_positive]
            + beta[mask_positive] * (std[mask_positive] / gamma[mask_positive])
        )

        w_fused[mask_negetive] = w[mask_negetive] * -1
        b_fused[mask_negetive] = (
            mean[mask_negetive]
            - b[mask_negetive]
            - beta[mask_negetive] * (std[mask_negetive] / gamma[mask_negetive])
        )
    # ******************* 普通bn融合 *******************
    else:
        w_fused = w * (gamma / std).reshape([conv.out_channels, 1, 1, 1])
        b_fused = beta + (b - mean) * (gamma / std)
    if bn_counter >= 2 and bn_counter <= bin_bn_fuse_num:
        bn_fused_conv = quantize.QuantConv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            W=args.W,
            quant_inference=True,
        )
    else:
        bn_fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
    bn_fused_conv.weight.data = w_fused
    bn_fused_conv.bias.data = b_fused
    return bn_fused_conv


def bn_fuse_module(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            conv_name_temp = name
            conv_child_temp = child
        elif isinstance(child, nn.BatchNorm2d):
            bn_fused_conv = bn_fuse(conv_child_temp, child)
            module._modules[conv_name_temp] = bn_fused_conv
            module._modules[name] = nn.Identity()
        else:
            bn_fuse_module(child)


def model_bn_fuse(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    bn_fuse_module(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", action="store", default="", help="gpu_id")
    parser.add_argument(
        "--prune_quant", action="store_true", help="this is prune_quant model"
    )
    parser.add_argument(
        "--model_type", type=int, default=1, help="model type:0-nin,1-nin_gc"
    )
    parser.add_argument("--W", type=int, default=2, help="Wb:2, Wt:3, Wfp:32")
    parser.add_argument("--A", type=int, default=2, help="Ab:2, Afp:32")

    args = parser.parse_args()
    print("==> Options:", args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.prune_quant:
        print("******Prune Quant model******")
        if args.model_type == 0:
            checkpoint = torch.load("../models_save/nin.pth")
            quant_model_train = nin.Net(cfg=checkpoint["cfg"])
        else:
            checkpoint = torch.load("../models_save/nin_gc.pth")
            quant_model_train = nin_gc.Net(cfg=checkpoint["cfg"])
    else:
        if args.model_type == 0:
            checkpoint = torch.load("../models_save/nin.pth")
            quant_model_train = nin.Net()
        else:
            checkpoint = torch.load("../models_save/nin_gc.pth")
            quant_model_train = nin_gc.Net()
    quant_bn_fused_model_inference = copy.deepcopy(quant_model_train)
    quantize.prepare(quant_model_train, inplace=True, A=args.A, W=args.W)
    quantize.prepare(
        quant_bn_fused_model_inference,
        inplace=True,
        A=args.A,
        W=args.W,
        quant_inference=True,
    )
    quant_model_train.load_state_dict(checkpoint["state_dict"])
    quant_bn_fused_model_inference.load_state_dict(checkpoint["state_dict"])

    # ********************** quant_model_train ************************
    torch.save(quant_model_train, "models_save/quant_model_train.pth")
    torch.save(quant_model_train.state_dict(), "models_save/quant_model_train_para.pth")
    model_array = np.array(quant_model_train)
    model_para_array = np.array(quant_model_train.state_dict())
    np.savetxt(
        "models_save/quant_model_train.txt", [model_array], fmt="%s", delimiter=","
    )
    np.savetxt(
        "models_save/quant_model_train_para.txt",
        [model_para_array],
        fmt="%s",
        delimiter=",",
    )

    # ********************* quant_bn_fused_model_inference **********************
    bn_counter = 0
    bin_bn_fuse_num = 0
    # 统计可以进行“针对特征(A)二值的BN融合”的BN层位置
    for m in quant_bn_fused_model_inference.modules():
        if isinstance(m, quantize.ActivationQuantizer):
            bin_bn_fuse_num += 1
    model_bn_fuse(quant_bn_fused_model_inference, inplace=True)  # bn融合
    print("***quant_model_train***\n", quant_model_train)
    print("\n***quant_bn_fused_model_inference***\n", quant_bn_fused_model_inference)
    torch.save(
        quant_bn_fused_model_inference, "models_save/quant_bn_fused_model_inference.pth"
    )
    torch.save(
        quant_bn_fused_model_inference.state_dict(),
        "models_save/quant_bn_fused_model_inference_para.pth",
    )
    model_array = np.array(quant_bn_fused_model_inference)
    model_para_array = np.array(quant_bn_fused_model_inference.state_dict())
    np.savetxt(
        "models_save/quant_bn_fused_model_inference.txt",
        [model_array],
        fmt="%s",
        delimiter=",",
    )
    np.savetxt(
        "models_save/quant_bn_fused_model_inference_para.txt",
        [model_para_array],
        fmt="%s",
        delimiter=",",
    )
    print("************* bn_fuse 完成 **************")
    print("************* bn_fused_model 已保存 **************")
