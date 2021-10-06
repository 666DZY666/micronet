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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", action="store", default="", help="gpu_id")
    parser.add_argument(
        "--prune_quant", action="store_true", help="this is prune_quant model"
    )
    parser.add_argument(
        "--model_type", type=int, default=1, help="model type:0-nin,1-nin_gc"
    )
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)

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
    quant_model_inference = copy.deepcopy(quant_model_train)
    quantize.prepare(
        quant_model_train, inplace=True, a_bits=args.a_bits, w_bits=args.w_bits
    )
    quantize.prepare(
        quant_model_inference,
        inplace=True,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        quant_inference=True,
    )
    quant_model_train.load_state_dict(checkpoint["state_dict"])
    quant_model_inference.load_state_dict(checkpoint["state_dict"])

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

    # ********************* quant_model_inference **********************
    print("***quant_model_train***\n", quant_model_train)
    print("\n***quant_model_inference***\n", quant_model_inference)
    torch.save(quant_model_inference, "models_save/quant_model_inference.pth")
    torch.save(
        quant_model_inference.state_dict(), "models_save/quant_model_inference_para.pth"
    )
    model_array = np.array(quant_model_inference)
    model_para_array = np.array(quant_model_inference.state_dict())
    np.savetxt(
        "models_save/quant_model_inference.txt", [model_array], fmt="%s", delimiter=","
    )
    np.savetxt(
        "models_save/quant_model_inference_para.txt",
        [model_para_array],
        fmt="%s",
        delimiter=",",
    )
    print("************* quant_model_para 已保存 **************")
