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

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def test():
    global best_acc
    model.eval()
    model_bn_merge.eval()
    test_loss = 0
    test_loss_bn_merge = 0
    correct = 0
    correct_bn_merge = 0

    for data, target in testloader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        output = model(data)
        output_bn_merge = model_bn_merge(data)

        test_loss += criterion(output, target).data.item()
        test_loss_bn_merge += criterion(output_bn_merge, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        pred_bn_merge = output_bn_merge.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        correct_bn_merge += pred_bn_merge.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * float(correct) / len(testloader.dataset)
    acc_bn_merge = 100. * float(correct_bn_merge) / len(testloader.dataset)
    test_loss /= len(testloader.dataset)
    test_loss_bn_merge /= len(testloader.dataset)

    print('\nmodel: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 256., correct, len(testloader.dataset),
        acc))
    print('\nmodel_bn_merge: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss_bn_merge * 256., correct_bn_merge, len(testloader.dataset),
        acc_bn_merge))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--gpu_id', action='store', default='',
            help='gpu_id')
    parser.add_argument('--data', action='store', default='../../data',
            help='dataset path')
    parser.add_argument('--eval_batch_size', type=int, default=128)
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

    model = torch.load('models_save/quan_model.pth')
    model_bn_merge = torch.load('models_save/quan_bn_merge_model.pth')
    if not args.cpu:
        model.cuda()
        model_bn_merge.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model_bn_merge = torch.nn.DataParallel(model_bn_merge, device_ids=range(torch.cuda.device_count()))

    criterion = nn.CrossEntropyLoss()
    
    print("*********bn_merge_test_model start*********")
    for epoch in range(1, args.epochs):
        test()
