from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import time
import nin_gc_inference
from bn_folding import DummyModule

# 量化模型测试
def test_quan_model():
    quan_model.eval()
    test_loss = 0
    average_test_loss = 0
    correct = 0
  
    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        output = quan_model(data)

        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    acc = 100. * float(correct) / len(testloader.dataset)
    average_test_loss = test_loss / (len(testloader.dataset) / args.eval_batch_size)

    print('\nquan_model: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}'.format(
        average_test_loss, correct, len(testloader.dataset), acc, inference_time * 1000, FPS))
    return

# 量化BN融合模型测试
def test_quan_bn_folding_model():
    quan_bn_folding_model.eval()
    test_loss_bn_folding = 0
    average_test_loss_bn_folding = 0
    correct_bn_folding = 0

    start_time = time.time()
    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        output_bn_folding = quan_bn_folding_model(data)

        test_loss_bn_folding += criterion(output_bn_folding, target).data.item()
        pred_bn_folding = output_bn_folding.data.max(1, keepdim=True)[1]
        correct_bn_folding += pred_bn_folding.eq(target.data.view_as(pred_bn_folding)).cpu().sum()
    end_time = time.time()
    inference_time = end_time - start_time
    FPS = len(testloader.dataset) / inference_time

    acc_bn_folding = 100. * float(correct_bn_folding) / len(testloader.dataset)
    average_test_loss_bn_folding = test_loss_bn_folding / (len(testloader.dataset) / args.eval_batch_size)

    print('\nquan_bn_folding_model: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), inference_time:{:.4f}ms, FPS:{:.4f}'.format(
        average_test_loss_bn_folding, correct_bn_folding, len(testloader.dataset), acc_bn_folding, inference_time * 1000, FPS))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--gpu_id', action='store', default='',
            help='gpu_id')
    parser.add_argument('--data', action='store', default='../../../data',
            help='dataset path')
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
            help='number of epochs to train (default: 160)')
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

    quan_model = nin_gc_inference.Net()
    quan_model.load_state_dict(torch.load('models_save/quan_model_para.pth'))   # 加载量化模型
    quan_bn_folding_model = torch.load('models_save/quan_bn_folding_model.pth') # 加载量化BN融合模型
    quan_model.eval()
    quan_bn_folding_model.eval()
    if not args.cpu:
        quan_model.cuda()
        quan_bn_folding_model.cuda()
        quan_model = torch.nn.DataParallel(quan_model, device_ids=range(torch.cuda.device_count()))
        quan_bn_folding_model = torch.nn.DataParallel(quan_bn_folding_model, device_ids=range(torch.cuda.device_count()))

    criterion = nn.CrossEntropyLoss()
    
    print("********* bn_folding_model_test start *********")
    # 融合前后模型对比测试,输出acc和FPS,由结果可知:BN融合成功,实现无损加速
    for epoch in range(1, args.epochs):
        test_quan_model()            # 量化模型测试
        test_quan_bn_folding_model() # 量化BN融合模型测试
