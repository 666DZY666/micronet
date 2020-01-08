#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from models import nin
import thop
from thop import profile
from models import nin_gc
from models import standard_dw

# 随机种子——训练结果可复现
def setup_seed(seed):
    # 为CPU设置种子用于生成随机数,以使得结果是确定的
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    # 为GPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed_all(seed)
    # 为numpy设置种子用于生成随机数,以使得结果是确定的
    np.random.seed(seed)
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层
    # 搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是
    # 网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，
    # 图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    # 反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间
    torch.backends.cudnn.deterministic = True

# 模型保存
def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    state_copy = state['state_dict'].copy()
    for key in state_copy.keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, args.save_path)
    #torch.save(state, 'models_save/nin_gc.pth')
    #torch.save(state, 'models_save/nin_preprune.pth')
    #torch.save(state, 'models_save/nin_gc_preprune.pth')
    #torch.save({'cfg': cfg, 'best_acc': best_acc, 'state_dict': state['state_dict']}, 'models_save/nin_refine.pth')
    #torch.save({'cfg': cfg, 'best_acc': best_acc, 'state_dict': state['state_dict']}, 'models_save/nin_gc_refine.pth')

#***********************稀疏训练（对BN层γ进行约束）**************************
def updateBN():
    for m in model.modules():
        #  isinstance() 函数来判断一个对象是否是一个已知的类型
        if isinstance(m, nn.BatchNorm2d):
            #  hasattr() 函数用于判断对象是否包含对应的属性
            if hasattr(m.weight, 'data'):
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data)) #L1正则

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):       
        data, target = Variable(data.cuda()), Variable(target.cuda())
        #print(data.size())
        # data shape: [50, 3, 32, 32]
        #print(target.size())
        # target shape: [50]
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        #***********************稀疏训练（对BN层γ进行约束）**************************
        if args.sr:
            updateBN()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 50., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [80, 130, 180, 230, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    # 使用argparse 的第一步是创建一个 ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 只有cpu可用的时候设置为true
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    # 设置网络用原始标准CNN，还是DepthWise CNN
    parser.add_argument('--type', default='0', type=int, help='use depthwise model')
    # 设置模型保存路径
    parser.add_argument('--save_path', default='models_save/nin.pth', type=str, help='mode save path')
    # 数据集路径
    parser.add_argument('--data', action='store', default='../data',
            help='dataset path')
    # 初始学习率
    parser.add_argument('--lr', action='store', default=0.01,
            help='the intial learning rate')
    # 权重惩罚项
    parser.add_argument('--wd', action='store', default=1e-7,
            help='nin_gc:0, nin:1e-5')
    # 添加模型从哪一轮开始继续训练
    parser.add_argument('--start_iter', default=0, type=int, help='the iter to start train the model')
    # 模型是否从之前保存的模型继续训练？
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='the path to the resume model')
    # 剪枝后的模型的保存路径
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
            help='the path to the refine(prune) model')
    # 测试模型
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    # sr(稀疏标志)
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
            help='train with channel sparsity regularization')
    # s(稀疏率)
    parser.add_argument('--s', type=float, default=0.0001,
            help='nin:0.0001, nin_gc:0.001')
    # 训练时的batch_size
    parser.add_argument('--train_batch_size', type=int, default=50)
    # 测试时的batch_size
    parser.add_argument('--eval_batch_size', type=int, default=50)
    # 线程数
    parser.add_argument('--num_workers', type=int, default=2)
    # 训练的epoch数
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
            help='number of epochs to train')
    args = parser.parse_args()
    print('==> Options:',args)

    # 设置随机数种子，结果可以复现
    setup_seed(1)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root = args.data, train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root = args.data, train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

    # 定义类别数组
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.refine:
        print('******Refine model******')
        #checkpoint = torch.load('models_save/nin_prune.pth')
        checkpoint = torch.load(args.refine)
        cfg = checkpoint['cfg']
        model = nin.Net(cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = 0
    else:
        print('******Initializing model******')
        model = nin.Net()
        if args.type == 0:
            model = nin.Net()
        elif args.type == 1:
            model = nin_gc.Net()
        elif args.type == 2:
            model = standard_dw.Net()
        '''
        cfg = []   #gc_prune —— cfg
        model = nin_gc.Net(cfg=cfg)
        '''
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    if args.resume:
        print('******Reume model******')
        #pretrained_model = torch.load('models_save/nin.pth')
        #pretrained_model = torch.load('models_save/nin_preprune.pth')
        #pretrained_model = torch.load('models_save/nin_refine.pth')
        pretrained_model = torch.load(args.resume)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print('***********************************Model**************************************')
    print(model)
    #input = torch.randn(1, 3, 32, 32)
    #flops, params = profile(model, inputs=(input,))
    #print('***********************************GFLOPs*************************************')
    #print(flops / 1024 / 1024 /1024)
    #print('***********************************Para(M)************************************')
    #print(params / 1024 / 1024)
    #print('***********************************End****************************************')

    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 'weight_decay':args.wd}]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=base_lr, weight_decay=args.wd)

    if args.evaluate:
        test()
        exit(0)
    # 学习率调整到start_iter这一轮的学习率
    for epoch in range(args.start_iter+1):
        adjust_learning_rate(optimizer, epoch)
    # 训练args.epochs这么多个epoch
    for epoch in range(args.start_iter, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
