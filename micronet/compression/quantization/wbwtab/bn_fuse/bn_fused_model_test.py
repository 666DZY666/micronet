from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../../../..")
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

# quant_model_train test
def test_quant_model_train():
    quant_model_train.eval()
    quant_test_loss = 0
    average_quant_test_loss = 0
    quant_correct = 0
  
    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        quant_output = quant_model_train(data)

        quant_test_loss += criterion(quant_output, target).data.item()
        quant_pred = quant_output.data.max(1, keepdim=True)[1]
        quant_correct += quant_pred.eq(target.data.view_as(quant_pred)).cpu().sum()
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    quant_acc = 100. * float(quant_correct) / len(testloader.dataset)
    average_quant_test_loss = quant_test_loss / (len(testloader.dataset) / args.eval_batch_size)

    print('\nquant_model_train: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}'.format(
        average_quant_test_loss, quant_correct, len(testloader.dataset), quant_acc, inference_time * 1000, FPS))
    return

# quant_bn_fused_model_inference test
def test_quant_bn_fused_model_inference():
    quant_bn_fused_model_inference.eval()
    quant_bn_fused_test_loss = 0
    average_quant_bn_fused_test_loss = 0
    quant_bn_fused_correct = 0

    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        quant_bn_fused_output = quant_bn_fused_model_inference(data)

        quant_bn_fused_test_loss += criterion(quant_bn_fused_output, target).data.item()
        quant_bn_fused_pred = quant_bn_fused_output.data.max(1, keepdim=True)[1]
        quant_bn_fused_correct += quant_bn_fused_pred.eq(target.data.view_as(quant_bn_fused_pred)).cpu().sum()
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    quant_bn_fused_acc = 100. * float(quant_bn_fused_correct) / len(testloader.dataset)
    average_quant_bn_fused_test_loss = quant_bn_fused_test_loss / (len(testloader.dataset) / args.eval_batch_size)

    print('quant_bn_fused_model_inference: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}'.format(
        average_quant_bn_fused_test_loss, quant_bn_fused_correct, len(testloader.dataset), quant_bn_fused_acc, inference_time * 1000, FPS))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--gpu_id', action='store', default='',
            help='gpu_id')
    parser.add_argument('--data', action='store', default='../../../../data',
            help='dataset path')
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
            help='number of epochs to train (default: 160)')
    # W —— 三值/二值(据训练时W量化(三/二值)情况而定)
    parser.add_argument('--W', type=int, default=2,
            help='Wb:2, Wt:3')
    args = parser.parse_args()
    print('==> Options:',args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root = args.data, train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # quant_model_train
    ori_model = torch.load('models_save/model.pth')                
    quant_model_train = quantize.prepare(ori_model, inplace=False, W=args.W, A=2)

    # quant_bn_fused_model_inference
    bn_fused_model = torch.load('models_save/bn_fused_model.pth')
    quant_bn_fused_model_inference = quantize.prepare(bn_fused_model, inplace=False, W=32, A=2)
    weight_quantizer = quantize.WeightQuantizer(W=args.W)
    for m in quant_bn_fused_model_inference.modules():
        if isinstance(m, quantize.QuantConv2d):
            m.weight.data = weight_quantizer(m.weight)

    # test
    if not args.cpu:
        quant_model_train.cuda()
        quant_bn_fused_model_inference.cuda()
        quant_model_train = torch.nn.DataParallel(quant_model_train, device_ids=range(torch.cuda.device_count()))
        quant_bn_fused_model_inference = torch.nn.DataParallel(quant_bn_fused_model_inference, device_ids=range(torch.cuda.device_count()))

    criterion = nn.CrossEntropyLoss()

    print("********* quant_bn_fused_model_inference test *********")
    for epoch in range(1, args.epochs):
        test_quant_model_train()            
        test_quant_bn_fused_model_inference() 
