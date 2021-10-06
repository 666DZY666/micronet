from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("..")
sys.path.append("../../../../..")
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import quantize


# quant_bn_fused_model_train test
def test_quant_bn_fused_model_train():
    quant_bn_fused_model_train_test_loss = 0
    quant_bn_fused_model_train_correct = 0

    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        quant_bn_fused_model_train_output = quant_bn_fused_model_train(data)

        quant_bn_fused_model_train_test_loss += criterion(
            quant_bn_fused_model_train_output, target
        ).data.item()
        quant_bn_fused_model_train_pred = quant_bn_fused_model_train_output.data.max(
            1, keepdim=True
        )[1]
        quant_bn_fused_model_train_correct += (
            quant_bn_fused_model_train_pred.eq(
                target.data.view_as(quant_bn_fused_model_train_pred)
            )
            .cpu()
            .sum()
        )
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    quant_bn_fused_model_train_acc = (
        100.0 * float(quant_bn_fused_model_train_correct) / len(testloader.dataset)
    )
    average_quant_bn_fused_model_train_test_loss = (
        quant_bn_fused_model_train_test_loss
        / (len(testloader.dataset) / args.eval_batch_size)
    )

    print(
        "\nquant_bn_fused_model_train: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}".format(
            average_quant_bn_fused_model_train_test_loss,
            quant_bn_fused_model_train_correct,
            len(testloader.dataset),
            quant_bn_fused_model_train_acc,
            inference_time * 1000,
            FPS,
        )
    )
    return


# quant_bn_fused_model_inference test
def test_quant_bn_fused_model_inference():
    quant_bn_fused_model_inference_test_loss = 0
    quant_bn_fused_model_inference_correct = 0

    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        quant_bn_fused_model_inference_output = quant_bn_fused_model_inference(data)

        quant_bn_fused_model_inference_test_loss += criterion(
            quant_bn_fused_model_inference_output, target
        ).data.item()
        quant_bn_fused_model_inference_pred = (
            quant_bn_fused_model_inference_output.data.max(1, keepdim=True)[1]
        )
        quant_bn_fused_model_inference_correct += (
            quant_bn_fused_model_inference_pred.eq(
                target.data.view_as(quant_bn_fused_model_inference_pred)
            )
            .cpu()
            .sum()
        )
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    quant_bn_fused_model_inference_acc = (
        100.0 * float(quant_bn_fused_model_inference_correct) / len(testloader.dataset)
    )
    average_quant_bn_fused_model_inference_test_loss = (
        quant_bn_fused_model_inference_test_loss
        / (len(testloader.dataset) / args.eval_batch_size)
    )

    print(
        "quant_bn_fused_model_inference: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}".format(
            average_quant_bn_fused_model_inference_test_loss,
            quant_bn_fused_model_inference_correct,
            len(testloader.dataset),
            quant_bn_fused_model_inference_acc,
            inference_time * 1000,
            FPS,
        )
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu", action="store_true", help="set if only CPU is available"
    )
    parser.add_argument("--gpu_id", action="store", default="", help="gpu_id")
    parser.add_argument(
        "--data", action="store", default="../../../../../data", help="dataset path"
    )
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 160)",
    )
    args = parser.parse_args()
    print("==> Options:", args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # quant_bn_fused_model_train
    quant_bn_fused_model_train = torch.load(
        "models_save/quant_bn_fused_model_train.pth"
    )
    quant_bn_fused_model_train.eval()
    # quant_bn_fused_model_inference
    quant_bn_fused_model_inference = torch.load(
        "models_save/quant_bn_fused_model_inference.pth"
    )
    quant_bn_fused_model_inference.eval()
    for m in quant_bn_fused_model_inference.modules():
        if isinstance(m, quantize.QuantConv2d):
            m.weight.data = m.weight_quantizer(m.weight)

    if not args.cpu:
        quant_bn_fused_model_train.cuda()
        quant_bn_fused_model_inference.cuda()

    criterion = nn.CrossEntropyLoss()
    print("********* quant_bn_fused_model_inference test *********")
    for epoch in range(1, args.epochs):
        test_quant_bn_fused_model_train()
        test_quant_bn_fused_model_inference()
