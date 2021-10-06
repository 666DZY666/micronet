import copy
import sys

sys.path.append("..")
sys.path.append("../../../../..")
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import nin_gc, nin

import quantize


# ******************** 是否保存模型完整参数 ********************
# torch.set_printoptions(precision=8, edgeitems=sys.maxsize, linewidth=200, sci_mode=False)


def bn_fuse(bn_conv):
    # ******************** bn参数 *********************
    mean = bn_conv.running_mean
    std = torch.sqrt(bn_conv.running_var + bn_conv.eps)
    gamma = bn_conv.gamma
    beta = bn_conv.beta
    # ******************* conv参数 ********************
    w = bn_conv.weight
    w_fused = w.clone()
    if bn_conv.bias is not None:
        b = bn_conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    b_fused = b.clone()
    # ******************* bn融合 *******************
    w_fused = w * (gamma / std).reshape([bn_conv.out_channels, 1, 1, 1])
    b_fused = beta + (b - mean) * (gamma / std)
    bn_fused_conv = quantize.QuantConv2d(
        bn_conv.in_channels,
        bn_conv.out_channels,
        bn_conv.kernel_size,
        stride=bn_conv.stride,
        padding=bn_conv.padding,
        dilation=bn_conv.dilation,
        groups=bn_conv.groups,
        bias=True,
        padding_mode=bn_conv.padding_mode,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        q_type=args.q_type,
        q_level=args.q_level,
        device=device,
        quant_inference=True,
    )
    bn_fused_conv.weight.data = w_fused
    bn_fused_conv.bias.data = b_fused
    bn_fused_conv.activation_quantizer.scale.copy_(bn_conv.activation_quantizer.scale)
    bn_fused_conv.activation_quantizer.zero_point.copy_(
        bn_conv.activation_quantizer.zero_point
    )
    bn_fused_conv.activation_quantizer.eps = bn_conv.activation_quantizer.eps
    bn_fused_conv.weight_quantizer.scale.copy_(bn_conv.weight_quantizer.scale)
    bn_fused_conv.weight_quantizer.zero_point.copy_(bn_conv.weight_quantizer.zero_point)
    bn_fused_conv.weight_quantizer.eps = bn_conv.weight_quantizer.eps
    return bn_fused_conv


def bn_fuse_module(module):
    for name, child in module.named_children():
        if isinstance(child, quantize.QuantBNFuseConv2d):
            bn_fused_conv = bn_fuse(child)
            module._modules[name] = bn_fused_conv
        else:
            bn_fuse_module(child)


def model_bn_fuse(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    bn_fuse_module(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu", action="store_true", help="set if only CPU is available"
    )
    parser.add_argument("--gpu_id", action="store", default="", help="gpu_id")
    parser.add_argument(
        "--prune_quant", action="store_true", help="this is prune_quant model"
    )
    parser.add_argument(
        "--model_type", type=int, default=1, help="model type:0-nin,1-nin_gc"
    )
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument(
        "--q_type", type=int, default=0, help="quant_type:0-symmetric, 1-asymmetric"
    )
    parser.add_argument(
        "--q_level", type=int, default=0, help="quant_level:0-per_channel, 1-per_layer"
    )
    args = parser.parse_args()
    print("==> Options:", args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.cpu:
        device = "cuda"
    else:
        device = "cpu"

    if args.prune_quant:
        print("******Prune Quant model******")
        if args.model_type == 0:
            checkpoint = torch.load("../models_save/nin_bn_fused.pth")
            quant_bn_fused_model_train = nin.Net(cfg=checkpoint["cfg"])
        else:
            checkpoint = torch.load("../models_save/nin_gc_bn_fused.pth")
            quant_bn_fused_model_train = nin_gc.Net(cfg=checkpoint["cfg"])
    else:
        if args.model_type == 0:
            checkpoint = torch.load("../models_save/nin_bn_fused.pth")
            quant_bn_fused_model_train = nin.Net()
        else:
            checkpoint = torch.load("../models_save/nin_gc_bn_fused.pth")
            quant_bn_fused_model_train = nin_gc.Net()
    quant_bn_fused_model_inference = copy.deepcopy(quant_bn_fused_model_train)
    quantize.prepare(
        quant_bn_fused_model_train,
        inplace=True,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        q_type=args.q_type,
        q_level=args.q_level,
        device=device,
        bn_fuse=1,
    )
    quantize.prepare(
        quant_bn_fused_model_inference,
        inplace=True,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        q_type=args.q_type,
        q_level=args.q_level,
        device=device,
        bn_fuse=1,
        quant_inference=True,
    )
    quant_bn_fused_model_train.load_state_dict(checkpoint["state_dict"])
    quant_bn_fused_model_inference.load_state_dict(checkpoint["state_dict"])

    # ********************** quant_bn_fused_model_train ************************
    torch.save(quant_bn_fused_model_train, "models_save/quant_bn_fused_model_train.pth")
    torch.save(
        quant_bn_fused_model_train.state_dict(),
        "models_save/quant_bn_fused_model_train_para.pth",
    )
    model_array = np.array(quant_bn_fused_model_train)
    model_para_array = np.array(quant_bn_fused_model_train.state_dict())
    np.savetxt(
        "models_save/quant_bn_fused_model_train.txt",
        [model_array],
        fmt="%s",
        delimiter=",",
    )
    np.savetxt(
        "models_save/quant_bn_fused_model_train_para.txt",
        [model_para_array],
        fmt="%s",
        delimiter=",",
    )

    # ********************* quant_bn_fused_model_inference **********************
    model_bn_fuse(quant_bn_fused_model_inference, inplace=True)  # bn融合
    print("***quant_bn_fused_model_train***\n", quant_bn_fused_model_train)
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
