# micronet

*"目前在深度学习领域分类两个派别，一派为学院派，研究强大、复杂的模型网络和实验方法，为了追求更高的性能；另一派为工程派，旨在将算法更稳定、高效的落地在硬件平台上，效率是其追求的目标。复杂的模型固然具有更好的性能，但是高额的存储空间、计算资源消耗是使其难以有效的应用在各硬件平台上的重要原因。所以，深度神经网络日益增长的规模为深度学习在移动端的部署带来了巨大的挑战，深度学习模型压缩与部署成为了学术界和工业界都重点关注的研究领域之一"*

## 项目简介

[![PyPI](https://img.shields.io/pypi/v/micronet)](https://pypi.org/project/micronet) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/micronet)

*micronet, a model compression and deploy lib.*

### 压缩

- 量化：QAT, High-Bit(>2b)、Low-Bit(≤2b)/Ternary and Binary; PTQ, 8-bit(tensorrt)
- 剪枝：正常、规整、分组卷积结构剪枝
- 针对特征(A)二值量化的BN融合
- High-Bit量化的BN融合

### 部署

- TensorRT


## 目前提供

- 普通卷积和分组卷积结构
- 权重W和特征A的训练中/后量化。训练中量化，W(32/16/8/4/2-bit, 三/二值) 和 A(32/16/8/4/2-bit, 二值)任意组合；训练后量化，采用tensorrt，支持8-bit
- 对称/非对称、per-layer/per-channel量化，MinMaxObserver/MovingAverageMinMaxObserver/HistogramObserver(calibration)
- 针对三/二值的一些tricks：W二值/三值缩放因子，W/grad（ste、saturate_ste、soft_ste）截断，A二值时采用B-A-C-P可比C-B-A-P获得更高acc等
- 多种剪枝方式：正常、规整（比如model可剪枝为每层剩余filter个数为N(8,16等)的倍数）、分组卷积结构（剪枝后仍保证分组卷积结构）的通道剪枝
- batch normalization融合及融合前后model对比测试：非量化普通BN融合（训练后，BN层参数 —> conv的权重w和偏置b）、针对特征(A)二值量化的BN融合（训练量化后，BN层参数 —> conv的偏置b)、High-Bit量化的BN融合（训练量化中，先融合再量化）
- 量化训练与量化推理仿真测试
- tensorrt：fp32/fp16/int8(ptq-calibration)、op-adapt(upsample)、dynamic_shape等


## 代码结构

![code_structure](https://github.com/666DZY666/micronet/blob/master/micronet/readme_imgs/code_structure.jpg)

```
micronet
├── __init__.py
├── compression
│   ├── README.md
│   ├── __init__.py
│   ├── pruning
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── gc_prune.py
│   │   ├── main.py
│   │   ├── models_save
│   │   │   └── models_save.txt
│   │   └── normal_regular_prune.py
│   └── quantization
│       ├── README.md
│       ├── __init__.py
│       ├── wbwtab
│       │   ├── __init__.py
│       │   ├── bn_fuse
│       │   │   ├── bn_fuse.py
│       │   │   ├── bn_fused_model_test.py
│       │   │   ├── models_save
│       │   │   │   └── models_save.txt
│       │   │   ├── nin_gc_inference.py
│       │   │   └── nin_gc_training.py
│       │   ├── main.py
│       │   ├── models_save
│       │   │   └── models_save.txt
│       │   └── quantize.py
│       └── wqaq
│           ├── __init__.py
│           ├── dorefa
│           │   ├── __init__.py
│           │   ├── main.py
│           │   ├── models_save
│           │   │   └── models_save.txt
│           │   └── quantize.py
│           └── iao
│               ├── __init__.py
│               ├── main.py
│               ├── models_save
│               │   └── models_save.txt
│               └── quantize.py
├── data
│   └── data.txt
├── deploy
│   ├── README.md
│   ├── __init__.py
│   └── tensorrt
│       ├── README.md
│       ├── __init__.py
│       ├── calibrator.py
│       ├── eval_trt.py
│       ├── models
│       │   ├── __init__.py
│       │   └── models_trt.py
│       ├── models_save
│       │   └── calibration_seg.cache
│       ├── test_trt.py
│       └── util_trt.py
├── models
│   ├── __init__.py
│   ├── nin.py
│   └── nin_gc.py
└── readme_imgs
    ├── code_structure.jpg
    └── micronet.xmind
```


## 项目进展
- **2019.12.4**, 初次提交
- **12.8**, DoReFa特征(A)量化前先进行缩放(* 0.1)，然后再截断，以减小截断误差
- **12.11**, 增加项目代码结构图
- 12.12, 完善使用示例
- 12.14, 增加:1、BN融合量化情况(W三值/二值)可选，即训练量化时选择W三/二值，这里则对应选择; 2、BN融合时对卷积核(conv)不带偏置(bias)的处理
- **12.17**, 增加模型压缩前后数据对比(示例)
- 12.20, 增加设备可选(cpu、gpu(单卡、多卡))
- **12.27**, 补充相关论文
- 12.29, 取消High-Bit量化8-bit以内的限制，即现在可以量化至10-bit、16-bit等
- **2020.2.17**, 1、精简W三值/二值量化代码; 2、加速W三值量化训练
- **2.18**, 优化针对特征(A)二值的BN融合:去除对BN层gamma参数的限制，即现在此情况下融合时BN可正常训练
- **2.24**, 再次优化三/二值量化代码组织结构，增强可移植性，旧版确实不太好移植。目前移植方法：将想要量化的Conv用compression/quantization/wbwtab/models/util_wbwtab.py中的QuantConv2d替换即可，可参照该路径下nin_gc.py中的使用方法
- **3.1**, 新增：1、google的High-Bit量化方法; 2、训练中High-Bit量化的BN融合
- **3.2、3.3**, 规整量化代码整体结构，目前所有量化方法都可采取类似的移植方式：将想要量化的Conv(或FC，目前dorefa支持，其他方法类似可写)用models/util_wxax.py中的QuantConv2d(或QuantLinear)替换即可，可分别参照该路径下nin_gc.py中的使用方法进行移植（分类、检测、分割等均适用，但需要据实际情况具体调试）
- **3.4**, 规整优化wbwtab/bn_fuse中“针对特征(A)二值的BN融合”的相关实现代码，可进行BN融合及融合前后模型对比测试(精度/速度/(大小))
- 3.11, 调整compression/wqaq/iao中的BN层momentum参数(0.1 —> 0.01),削弱batch统计参数占比,一定程度抑制量化带来的抖动。经实验,量化训练更稳定,acc提升1%左右
- **3.13**, 更新代码结构图
- 4.6, 修正二值量化训练中W_clip的相关问题(之前由于这个，导致二值量化训练精度上不去，现在已可正常使用)(同时修正无法找到一些模块如models/util_wxax.py的问题)
- **12.14**, 1、improve code structure; 2、add deploy-tensorrt(main module, but not running yet)
- 12.18, 1、improve code structure/module reference/module_name; 2、add transfer-use demo
- **12.21**, improve pruning-quantization pipeline and code
- **2021.1.4**, add other quant_op
- 1.5, add quant_weight's per-channel and per-layer selection
- **1.7**, fix iao's loss-nan bug. The bug is due to per-channel min/max error
- 1.8, 1、improve quant_para save. Now, only save scale and zero_point; 2、add optional weight_observer(MinMaxObserver or MovingAverageMinMaxObserver)
- **1.11**, fix bug in binary_a(1/0) and binary_w preprocessing
- **1.12**, add "pip install"
- **1.22**, add auto_insert_quant_op(this still needs to be improved)
- **1.27**, improve auto_insert_quant_op(now you can easily use quantization, as [quant_test_auto](#quant_test_auto.py))
- 1.28, 1、fix prune-quantization pipeline and code; 2、improve code structure
- **2.1**, improve wbwtab_bn_fuse
- **2.4**, 1、add wqaq_bn_fuse; 2、add quant_model_inference_simulation


## 环境要求

- python >= 3.5
- torch >= 1.1.0
- torchvison >= 0.3.0
- numpy
- onnx == 1.6.0
- tensorrt == 7.0.0.11


## 安装

[PyPI](https://pypi.org/project/micronet/)

```bash
pip install micronet -i https://pypi.org/simple
```

[GitHub](https://github.com/666DZY666/micronet)

```bash
git clone https://github.com/666DZY666/micronet.git
cd micronet
python setup.py install
```

*验证*

```bash
python -c "import micronet; print(micronet.__version__)"
```

## 测试

*Install from github*

### 压缩

#### 量化

*--refine,可加载预训练浮点模型参数,在其基础上做量化*

##### wbwtab

--W --A, 权重W和特征A量化取值

```bash
cd micronet/compression/quantization/wbwtab
```

- WbAb

```bash
python main.py --W 2 --A 2
```

- WbA32

```bash
python main.py --W 2 --A 32
```

- WtAb

```bash
python main.py --W 3 --A 2
```

- WtA32

```bash
python main.py --W 3 --A 32
```

##### wqaq

--w_bits --a_bits, 权重W和特征A量化位数

###### dorefa

```bash
cd micronet/compression/quantization/wqaq/dorefa
```

- W16A16

```bash
python main.py --w_bits 16 --a_bits 16
```

- W8A8

```bash
python main.py --w_bits 8 --a_bits 8
```

- W4A4

```bash
python main.py --w_bits 4 --a_bits 4
```

- 其他bits情况类比

###### iao

```bash
cd micronet/compression/quantization/wqaq/iao
```

*量化位数选择同dorefa*

*单卡*

--q_type, 量化类型(0-对称, 1-非对称)

--q_level, 权重量化级别(0-通道级, 1-层级)

--bn_fuse, 量化中bn融合标志(0-不融合, 1-融合)

--weight_observer, weight_observer选择(0-MinMaxObserver, 1-MovingAverageMinMaxObserver)

- (默认)对称、(权重)通道级量化, bn不融合, weight_observer-MinMaxObserver

```bash
python main.py --q_type 0 --q_level 0 --bn_fuse 0 --weight_observer 0 --gpu_id 0
```

- 对称、(权重)通道级量化, bn不融合, weight_observer-MovingAverageMinMaxObserver

```bash
python main.py --q_type 0 --q_level 0 --bn_fuse 0 --weight_observer 1 --gpu_id 0
```

- 对称、(权重)层级量化, bn不融合

```bash
python main.py --q_type 0 --q_level 1 --bn_fuse 0 --gpu_id 0
```

- 非对称、(权重)通道级量化, bn不融合

```bash
python main.py --q_type 1 --q_level 0 --bn_fuse 0 --gpu_id 0
```

- 非对称、(权重)层级量化, bn不融合

```bash
python main.py --q_type 1 --q_level 1 --bn_fuse 0 --gpu_id 0
```

- 对称、(权重)通道级量化, bn融合

```bash
python main.py --q_type 0 --q_level 0 --bn_fuse 1 --gpu_id 0
```

- 对称、(权重)层级量化, bn融合

```bash
python main.py --q_type 0 --q_level 1 --bn_fuse 1 --gpu_id 0
```

- 非对称、(权重)通道级量化, bn融合

```bash
python main.py --q_type 1 --q_level 0 --bn_fuse 1 --gpu_id 0
```

- 非对称、(权重)层级量化, bn融合

```bash
python main.py --q_type 1 --q_level 1 --bn_fuse 1 --gpu_id 0
```


#### 剪枝

*稀疏训练  —>  剪枝  —>  微调*

```bash
cd micronet/compression/pruning
```

##### 稀疏训练

-sr 稀疏标志

--s 稀疏率(需根据dataset、model情况具体调整)

--model_type 模型类型(0-nin, 1-nin_gc)

- nin(正常卷积结构)

```bash
python main.py -sr --s 0.0001 --model_type 0
```

- nin_gc(含分组卷积结构)

```bash
python main.py -sr --s 0.001 --model_type 1
```

##### 剪枝

--percent 剪枝率

--normal_regular 正常、规整剪枝标志及规整剪枝基数(如设置为N,则剪枝后模型每层filter个数即为N的倍数)

--model 稀疏训练后的model路径

--save 剪枝后保存的model路径（路径默认已给出, 可据实际情况更改）

- 正常剪枝(nin)

```bash
python normal_regular_prune.py --percent 0.5 --model models_save/nin_sparse.pth --save models_save/nin_prune.pth
```

- 规整剪枝(nin)

```bash
python normal_regular_prune.py --percent 0.5 --normal_regular 8 --model models_save/nin_sparse.pth --save models_save/nin_prune.pth
```

或

```bash
python normal_regular_prune.py --percent 0.5 --normal_regular 16 --model models_save/nin_sparse.pth --save models_save/nin_prune.pth
```

- 分组卷积结构剪枝(nin_gc)

```bash
python gc_prune.py --percent 0.4 --model models_save/nin_gc_sparse.pth
```

##### 微调

--prune_refine 剪枝后的model路径（在其基础上做微调）

- nin

```bash
python main.py --model_type 0 --prune_refine models_save/nin_prune.pth
```

- nin_gc

*需要传入**剪枝**后得到的新模型的**cfg***

*如*

```bash
python main.py --model_type 1 --gc_prune_refine 154 162 144 304 320 320 608 584
```

#### 剪枝 —> 量化（注意剪枝率和量化率平衡）

*剪枝完成后，加载保存的模型参数在其基础上再做量化*

##### 剪枝 —> 量化（高位）（剪枝率偏大、量化率偏小）

- dorefa

```bash
cd micronet/compression/quantization/wqaq/dorefa
```

或 

- iao

*单卡*

```bash
cd micronet/compression/quantization/wqaq/iao
```

###### w8a8

- nin(正常卷积结构)

```bash
python main.py --w_bits 8 --a_bits 8 --model_type 0 --prune_refine ../../../pruning/models_save/nin_finetune.pth
```

- nin_gc(含分组卷积结构)

```bash
python main.py --w_bits 8 --a_bits 8 --model_type 1 --prune_refine ../../../pruning/models_save/nin_gc_retrain.pth
```

###### 其他可选量化配置类比

##### 剪枝 —> 量化（低位）（剪枝率偏小、量化率偏大）

```bash
cd micronet/compression/quantization/wbwtab
```

###### wbab

- nin(正常卷积结构)

```bash
python main.py --W 2 --A 2 --model_type 0 --prune_refine ../../pruning/models_save/nin_finetune.pth
```

- nin_gc(含分组卷积结构)

```bash
python main.py --W 2 --A 2 --model_type 1 --prune_refine ../../pruning/models_save/nin_gc_retrain.pth
```

###### 其他取值情况类比


#### BN融合与量化推理仿真测试

##### wbwtab

```bash
cd micronet/compression/quantization/wbwtab/bn_fuse
```

###### bn_fuse(得到quant_model_train和quant_bn_fused_model_inference的结构和参数)

*--model_type, 1 - nin_gc(含分组卷积结构); 0 - nin(正常卷积结构)*

*--prune_quant, 剪枝_量化模型标志*

*--W, weight量化取值*

*均需要与量化训练保持一致,可直接用默认*

- nin_gc, quant_model, wb

```bash
python bn_fuse.py --model_type 1 --W 2
```

- nin_gc, prune_quant_model, wb

```bash
python bn_fuse.py --model_type 1 --prune_quant --W 2
```

- nin_gc, quant_model, wt

```bash
python bn_fuse.py --model_type 1 --W 3
```

- nin, quant_model, wb

```bash
python bn_fuse.py --model_type 0 --W 2
```

###### bn_fused_model_test(对quant_model_train和quant_bn_fused_model_inference进行测试)

```bash
python bn_fused_model_test.py
```

##### dorefa

```bash
cd micronet/compression/quantization/wqaq/dorefa/quant_model_test
```

###### quant_model_para(得到quant_model_train和quant_model_inference的结构和参数)

*--model_type, 1 - nin_gc(含分组卷积结构); 0 - nin(正常卷积结构)*

*--prune_quant, 剪枝_量化模型标志*

*--w_bits, weight量化位数; --a_bits, activation量化位数*

*均需要与量化训练保持一致,可直接用默认*

- nin_gc, quant_model, w8a8

```bash
python quant_model_para.py --model_type 1 --w_bits 8 --a_bits 8
```

- nin_gc, prune_quant_model, w8a8

```bash
python quant_model_para.py --model_type 1 --prune_quant --w_bits 8 --a_bits 8
```

- nin, quant_model, w8a8

```bash
python quant_model_para.py --model_type 0 --w_bits 8 --a_bits 8
```

###### quant_model_test(对quant_model_train和quant_model_inference进行测试)

```bash
python quant_model_test.py
```

##### iao

***注意, 量化训练时 --bn_fuse 需要设置为1***
```bash
cd micronet/compression/quantization/wqaq/iao/bn_fuse
```

###### bn_fuse(得到quant_bn_fused_model_train和quant_bn_fused_model_inference的结构和参数)

*--model_type, 1 - nin_gc(含分组卷积结构); 0 - nin(正常卷积结构)*

*--prune_quant, 剪枝_量化模型标志*

*--w_bits, weight量化位数; --a_bits, activation量化位数*

*--q_type, 0 - 对称; 1 - 非对称*

*--q_level, 0 - 通道级; 1 - 层级*

*均需要与量化训练保持一致,可直接用默认*

- nin_gc, quant_model, w8a8

```bash
python quant_model_para.py --model_type 1 --w_bits 8 --a_bits 8
```

- nin_gc, prune_quant_model, w8a8

```bash
python quant_model_para.py --model_type 1 --prune_quant --w_bits 8 --a_bits 8
```

- nin, quant_model, w8a8

```bash
python quant_model_para.py --model_type 0 --w_bits 8 --a_bits 8
```

- nin_gc, quant_model, w8a8, 非对称, 层级

```bash
python quant_model_para.py --model_type 0 --w_bits 8 --a_bits 8 --q_type 1 --q_level 1
```

###### bn_fused_model_test(对quant_bn_fused_model_train和quant_bn_fused_model_inference进行测试)

```bash
python bn_fused_model_test.py
```

#### 设备选取

*现支持cpu、gpu(单卡、多卡)*

--cpu 使用cpu，--gpu_id 使用并选择gpu

- cpu

```bash
python main.py --cpu
```

- gpu单卡

```bash
python main.py --gpu_id 0
```

或

```bash
python main.py --gpu_id 1
```

- gpu多卡

```bash
python main.py --gpu_id 0,1
```

或

```bash
python main.py --gpu_id 0,1,2
```

*默认，使用服务器全卡*

### 部署

#### TensorRT

*目前仅提供相关**核心模块**代码，后续再加入完整可运行demo*

##### 相关解读
- [tensorrt-基础](https://zhuanlan.zhihu.com/p/336256668)
- [tensorrt-op/dynamic_shape](https://zhuanlan.zhihu.com/p/335829625)


## 迁移

### 量化训练

#### LeNet example

##### quant_test_manual.py

*A model can be quantized(High-Bit(>2b)、Low-Bit(≤2b)/Ternary and Binary) by simply replacing ***op*** with ***quant_op***.*

```python
import torch.nn as nn
import torch.nn.functional as F

# ``quantize`` is quant_module, ``QuantConv2d``, ``QuantLinear``, ``QuantMaxPool2d``, ``QuantReLU`` are quant_op
from micronet.compression.quantization.wbwtab.quantize import QuantConv2d as quant_conv_wbwtab
from micronet.compression.quantization.wbwtab.quantize import ActivationQuantizer as quant_relu_wbwtab
from micronet.compression.quantization.wqaq.dorefa.quantize import QuantConv2d as quant_conv_dorefa
from micronet.compression.quantization.wqaq.dorefa.quantize import QuantLinear as quant_linear_dorefa
from micronet.compression.quantization.wqaq.iao.quantize import QuantConv2d as quant_conv_iao
from micronet.compression.quantization.wqaq.iao.quantize import QuantLinear as quant_linear_iao
from micronet.compression.quantization.wqaq.iao.quantize import QuantMaxPool2d as quant_max_pool_iao
from micronet.compression.quantization.wqaq.iao.quantize import QuantReLU as quant_relu_iao

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class QuantLeNetWbWtAb(nn.Module):
    def __init__(self):
        super(QuantLeNetWbWtAb, self).__init__()
        self.conv1 = quant_conv_wbwtab(1, 10, kernel_size=5)
        self.conv2 = quant_conv_wbwtab(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = quant_relu_wbwtab()

    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class QuantLeNetDoReFa(nn.Module):
    def __init__(self):
        super(QuantLeNetDoReFa, self).__init__()
        self.conv1 = quant_conv_dorefa(1, 10, kernel_size=5)
        self.conv2 = quant_conv_dorefa(10, 20, kernel_size=5)
        self.fc1 = quant_linear_dorefa(320, 50)
        self.fc2 = quant_linear_dorefa(50, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class QuantLeNetIAO(nn.Module):
    def __init__(self):
        super(QuantLeNetIAO, self).__init__()
        self.conv1 = quant_conv_iao(1, 10, kernel_size=5)
        self.conv2 = quant_conv_iao(10, 20, kernel_size=5)
        self.fc1 = quant_linear_iao(320, 50)
        self.fc2 = quant_linear_iao(50, 10)
        self.max_pool = quant_max_pool_iao(kernel_size=2)
        self.relu = quant_relu_iao(inplace=True)

    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

lenet = LeNet()
quant_lenet_wbwtab = QuantLeNetWbWtAb()
quant_lenet_dorefa = QuantLeNetDoReFa()
quant_lenet_iao = QuantLeNetIAO()

print('***ori_model***\n', lenet)
print('\n***quant_model_wbwtab***\n', quant_lenet_wbwtab)
print('\n***quant_model_dorefa***\n', quant_lenet_dorefa)
print('\n***quant_model_iao***\n', quant_lenet_iao)

print('\nquant_model is ready')
print('micronet is ready')
```

##### quant_test_auto.py

*A model can be quantized(High-Bit(>2b)、Low-Bit(≤2b)/Ternary and Binary) by simply using ***micronet.compression.quantization.quantize.prepare(model)***.*

```python
import torch.nn as nn
import torch.nn.functional as F

import micronet.compression.quantization.wqaq.dorefa.quantize as quant_dorefa
import micronet.compression.quantization.wqaq.iao.quantize as quant_iao

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))
        x = self.relu(self.max_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

lenet = LeNet()
quant_lenet_dorefa = quant_dorefa.prepare(lenet, inplace=False)
quant_lenet_iao = quant_iao.prepare(lenet, inplace=False)

print('***ori_model***\n', lenet)
print('\n***quant_model_dorefa***\n', quant_lenet_dorefa)
print('\n***quant_model_iao***\n', quant_lenet_iao)

print('\nquant_model is ready')
print('micronet is ready')
```

#### test

##### quant_test_manual

```bash
python -c "import micronet; micronet.quant_test_manual()"
```

##### quant_test_auto

```bash
python -c "import micronet; micronet.quant_test_auto()"
```

*when outputting "quant_model is ready", micronet is ready.*

### 量化推理

***参考[BN融合与量化推理仿真测试](#bn融合与量化推理仿真测试)***

## 模型压缩数据对比（仅供参考）

*以下为cifar10示例，可在更冗余模型、更大数据集上尝试其他组合压缩方式*

|类型|W(Bits)|A(Bits)|Acc|GFLOPs|Para(M)|Size(MB)|压缩率|损失|
|:-:|:-----:|:-----:|:--:|:---:|:-----:|:------:|:---:|:-:|
|原模型(nin)|FP32|FP32|91.01%|0.15|0.67|2.68|***|***|
|采用分组卷积结构(nin_gc)|FP32|FP32|91.04%|0.15|0.58|2.32|13.43%|-0.03%|
|剪枝|FP32|FP32|90.26%|0.09|0.32|1.28|52.24%|0.75%|
|量化|1|FP32|90.93%|***|0.58|0.204|92.39%|0.08%|
|量化|1.5|FP32|91%|***|0.58|0.272|89.85%|0.01%|
|量化|1|1|86.23%|***|0.58|0.204|92.39%|4.78%|
|量化|1.5|1|86.48%|***|0.58|0.272|89.85%|4.53%|
|量化(DoReFa)|8|8|91.03%|***|0.58|0.596|77.76%|-0.02%|
|量化(IAO,全量化,symmetric/per-channel/bn_fuse)|8|8|90.99%|***|0.58|0.596|77.76%|0.02%|
|分组+剪枝+量化|1.5|1|86.13%|***|0.32|0.19|92.91%|4.88%|

*--train_batch_size 256, 单卡*

## 相关资料

### 压缩

#### 量化

##### QAT

###### 二值

- [BinarizedNeuralNetworks: TrainingNeuralNetworkswithWeightsand ActivationsConstrainedto +1 or−1](https://arxiv.org/abs/1602.02830)

- [XNOR-Net:ImageNetClassiﬁcationUsingBinary ConvolutionalNeuralNetworks](https://arxiv.org/abs/1603.05279)

- [AN EMPIRICAL STUDY OF BINARY NEURAL NETWORKS’ OPTIMISATION](https://openreview.net/forum?id=rJfUCoR5KX)

- [A Review of Binarized Neural Networks](https://www.semanticscholar.org/paper/A-Review-of-Binarized-Neural-Networks-Simons-Lee/0332fdf00d7ff988c5b66c47afd49431eafa6cd1)

###### 三值

- [Ternary weight networks](https://arxiv.org/abs/1605.04711)

###### High-Bit

- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)

##### PTQ

###### High-Bit

- [tensorrt-ptq-8-bit](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)

#### 剪枝

- [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)
- [RETHINKING THE VALUE OF NETWORK PRUNING](https://arxiv.org/abs/1810.05270)

#### 适配专用芯片的模型压缩

- [Convolutional Networks for Fast, Energy-Efficient Neuromorphic Computing](https://arxiv.org/abs/1603.08270)

### 部署

#### TensorRT

- [github](https://github.com/NVIDIA/TensorRT)
- [ptq](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
- [tensorrt-基础](https://zhuanlan.zhihu.com/p/336256668)
- [tensorrt-op/dynamic_shape](https://zhuanlan.zhihu.com/p/335829625)
- [summary](https://github.com/mileistone/study_resources/blob/master/engineering/tensorrt/tensorrt.md)


## 后续

- tensorrt完整demo
- 其他压缩算法(量化/剪枝/蒸馏/NAS等)
- 其他部署框架(mnn/tnn/tengine等)
- 压缩 —> 部署