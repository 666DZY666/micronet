from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")
import math
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import init
from models import nin_gc, nin

def setup_seed(seed):
    torch.manual_seed(seed)                    
    #torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    np.random.seed(seed)                       
    torch.backends.cudnn.deterministic = True

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
    if args.model_type == 0:
        if args.sr:
            torch.save(state, 'models_save/nin_sparse.pth')
        elif args.prune_refine:
            torch.save({'cfg': cfg, 'best_acc': best_acc, 'state_dict': state['state_dict']}, 'models_save/nin_finetune.pth')
        else:
            torch.save(state, 'models_save/nin.pth')
    else:
        if args.sr:
            torch.save(state, 'models_save/nin_gc_sparse.pth')
        elif args.gc_prune_refine:
            torch.save({'cfg': cfg, 'best_acc': best_acc, 'state_dict': state['state_dict']}, 'models_save/nin_gc_retrain.pth')
        else:
            torch.save(state, 'models_save/nin_gc.pth')
        
#***********************稀疏训练（对BN层γ进行约束）**************************
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if hasattr(m.weight, 'data'):
                m.weight.grad.data.add_(args.s * torch.sign(m.weight.data)) # L1正则

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(trainloader):       
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
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
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)             
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
        test_loss * 256., correct, len(testloader.dataset),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    # gpu_id
    parser.add_argument('--gpu_id', action='store', default='',
            help='gpu_id')
    parser.add_argument('--data', action='store', default='../../data',
            help='dataset path')
    parser.add_argument('--lr', action='store', default=0.01,
            help='the intial learning rate')
    parser.add_argument('--wd', action='store', default=1e-7,
            help='nin_gc:0, nin:1e-5')
    # prune_refine
    parser.add_argument('--prune_refine', default='', type=str, metavar='PATH',
            help='the path to the prune_refine model')
    # refine
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
            help='the path to the float_refine model')
    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='the path to the resume model')
    # gc_prune_refine的cfg
    parser.add_argument('--gc_prune_refine', nargs='+', type=int,
            help='gc_prune_refine-cfg')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--train_batch_size', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
            help='number of epochs to train')
    # sr(稀疏标志)
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
            help='train with channel sparsity regularization')
    # s(稀疏率)
    parser.add_argument('--s', type=float, default=0.0001,
            help='nin:0.0001, nin_gc:0.001')
    # 模型结构选择
    parser.add_argument('--model_type', type=int, default=1,
            help='model type:0-nin,1-nin_gc')   
    args = parser.parse_args()
    print('==> Options:',args)

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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

    # define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.prune_refine:
        print('******Prune Refine model******')
        #checkpoint = torch.load('models_save/nin_prune.pth')
        checkpoint = torch.load(args.prune_refine)
        cfg = checkpoint['cfg']
        model = nin.Net(cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = 0
    elif args.refine:
        print('******Float Refine model******')
        #checkpoint = torch.load('models_save/nin.pth')
        state_dict = torch.load(args.refine)
        if args.model_type == 0:
            model = nin.Net()
        else:
            model = nin_gc.Net()
        model.load_state_dict(state_dict)
        best_acc = 0
    elif args.resume:
        print('******Reume model******')
        #checkpoint = torch.load('models_save/nin.pth')
        #checkpoint = torch.load('models_save/nin_sparse.pth')
        checkpoint = torch.load(args.resume)
        if args.model_type == 0:
            model = nin.Net()
        else:
            model = nin_gc.Net()
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
    else:
        # nin_gc_retrain
        if args.gc_prune_refine:
            print('******GCPrune Refine model******')
            cfg = args.gc_prune_refine  
            model = nin_gc.Net(cfg=cfg)
        else:
            print('******Initializing model******')
            if args.model_type == 0:
                model = nin.Net()
            else:
                model = nin_gc.Net()
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

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

    for epoch in range(1, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
