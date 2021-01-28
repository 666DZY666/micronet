import sys
sys.path.append("../..")
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import nin_gc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', default='../../data',
        help='dataset path')
parser.add_argument('--cpu', action='store_true',
        help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.4,
        help='nin_gc:0.4')
parser.add_argument('--layers', type=int, default=9,
        help='layers (default: 9)')
parser.add_argument('--model', default='models_save/nin_gc_preprune.pth', type=str, metavar='PATH',
        help='path to raw trained model (default: none)')
args = parser.parse_args()
layers = args.layers
print(args)

model = nin_gc.Net()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        model.load_state_dict(torch.load(args.model)['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print('旧模型: ', model)
total = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
y, j = torch.sort(bn)
thre_index = int(total * args.percent)
if thre_index == total:
    thre_index = total - 1
thre_0 = y[thre_index]

#****************获取针对分组卷积剪枝每层的基数（base_bumber）********************
nums = []
channels = []
groups = [1]
prune_base_num = []
j = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        s_0 = m.weight.data.data.size()[0]
        s_1 = m.weight.data.data.size()[1]
        nums.append(s_0)
        channels.append(s_1)
while  j < len(nums) - 1:
    groups.append(int(nums[j] / channels[j+1])) 
    j += 1
#print(groups)
j = 0
while  j < len(groups) - 1:
    for i in range(1, (groups[j] * groups[j+1])+1):
        if i % groups[j] == 0 and i % groups[j+1] == 0:
            prune_base_num.append(i)       
            break
    j += 1
#print(prune_base_num)

#********************************预剪枝*********************************
pruned = 0
cfg_0 = []
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre_0).float()
            remain_channels = torch.sum(mask)

            if remain_channels == 0:
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))]=1

            # ******************分组卷积剪枝******************
            base_number = prune_base_num[i-1]
            v = 0
            n = 1
            if remain_channels % base_number != 0:
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)
                        
                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_0.append(mask.shape[0])
            cfg.append(int(remain_channels))
            cfg_mask.append(mask.clone())
            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0]))
pruned_ratio = float(pruned/total)
print('\r\n!预剪枝完成!')
print('total_pruned_ratio: ', pruned_ratio)
#********************************预剪枝后model测试*********************************
def test():
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root = args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size = 64, shuffle=False, num_workers=1)
    model.eval()
    correct = 0
    
    for data, target in test_loader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Accuracy: {:.2f}%\n'.format(acc))
    return
print('************预剪枝模型测试************')
if not args.cpu:
    model.cuda()
test()

#*****************************剪枝后model结构********************************
newmodel = nin_gc.Net(cfg)
print('新模型: ', newmodel)

#*****************************剪枝前后model对比********************************
print('************旧模型结构************')
print(cfg_0)
print('************新模型结构************')
print(cfg, '\r\n')
