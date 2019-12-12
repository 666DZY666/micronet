# model-compression
"目前在深度学习领域分类两个派别，一派为学院派，研究强大、复杂的模型网络和实验方法，为了追求更高的性能；另一派为工程派，旨在将算法更稳定、高效的落地在硬件平台上，效率是其追求的目标。复杂的模型固然具有更好的性能，但是高额的存储空间、计算资源消耗是使其难以有效的应用在各硬件平台上的重要原因。所以，卷积神经网络日益增长的深度和尺寸为深度学习在移动端的部署带来了巨大的挑战，深度学习模型压缩与加速成为了学术界和工业界都重点关注的研究领域之一"

## 项目简介 
基于pytorch实现模型压缩（1、量化：8/4/2 bits(dorefa)、三值/二值(twn/bnn/xnor-net)；2、剪枝：正常、规整、针对分组卷积结构的通道剪枝；3、分组卷积结构；4、针对特征A二值的BN融合）

## 目前提供
- 1、普通卷积和分组卷积结构
- 2、权重W和特征A的训练中量化，W（32/8/4/2bits，三/二值） 和A（32/8/4/2bits，三/二值）任意组合
- 3、针对三/二值的一些tricks：W二值/三值缩放因子，W/grad（ste、saturate_ste、soft_ste）截断，W三值_gap(防止参数更新抖动)，W/A二值时BN_momentum(<0.9)，A二值时采用B-A-C-P可比C-B-A-P获得更高acc
- 4、多种剪枝方式：正常剪枝、规整剪枝（比如可剪枝为剩余filter个数为16的倍数）、针对分组卷积结构的剪枝（剪枝后仍保证分组卷积结构）
- 5、batch normalization的融合及融合前后model对比测试：普通融合（BN层参数 —> conv的权重w和偏置b）、针对特征A二值的融合（BN层参数 —> conv的偏置b)

## 代码结构
![img1](https://github.com/666DZY666/model-compression/blob/master/readme_imgs/code_structure.jpg)

## 使用
### 量化
#### W（FP32/三/二值）、A（FP32/三/二值）
--W --A, 权重W和特征A量化取值
```
cd WbWtAb
```
- WbAb
```
python main.py --W 2 --A 2
```
- WbA32
```
python main.py --W 2 --A 32
```
- WtAb
```
python main.py --W 3 --A 2
```
- WtA32
```
python main.py --W 3 --A 32
```
#### W（FP32/8/4/2 bits）、A（FP32/8/4/2 bits）
--Wbits --Abits, 权重W和特征A量化位数
```
cd WqAq
```
- W8A8
```
python main.py --Wbits 8 --Abits 8
```
- W4A8
```
python main.py --Wbits 4 --Abits 8
```
- W4A4
```
python main.py --Wbits 4 --Abits 4
```
- 其他情况类比
### 剪枝
```
cd prune
```
#### 正常训练
```
python main.py
```
#### 稀疏训练
-sr 稀疏标志, --s 稀疏率(需根据dataset、model情况具体调整)
- nin(正常卷积结构)
```
python main.py -sr --s 0.0001
```
- nin_gc(含分组卷积结构)
```
python main.py -sr --s 0.001
```
#### 剪枝
--percent 剪枝率, --normal_regular 正常、规整剪枝标志及规整剪枝基数(如设置为N,则剪枝后模型每层filter个数即为N的倍数), --model 稀疏训练后的model路径, --save 剪枝后保存的model路径（路径默认已给出, 可据实际情况更改）
- 正常剪枝
```
python normal_regular_prune.py --percent 0.5
```
- 规整剪枝
```
python normal_regular_prune.py --percent 0.5 --normal_regular 8
```
```
python normal_regular_prune.py --percent 0.5 --normal_regular 16
```
- 分组卷积结构剪枝
```
python gc_prune.py --percent 0.4
```
#### 微调
--refine 剪枝后的model路径（在其基础上做微调）
```
python main.py --refine models_save/nin_preprune.pth
```
### 剪枝 —> 量化（注意剪枝率和量化率平衡）
#### 待补充。。。
### 分组+剪枝 —> 量化（注意剪枝率和量化率平衡）
#### 待补充。。。
### batch normalization融合及融合前后测试
#### 待补充。。。

## 可尝试
- 1、渐进式量化：FP32—>8—>4—>三/二值
- 2、对冗余度更高的模型做压缩

## 后续补充
- 1、使用示例及说明
- 2、模型压缩前后详细数据对比
- 3、参考论文及工程
- 4、imagenet测试（目前cifar10）

## 后续扩充
- 1、Nvidia、Google的INT8量化方案
- 2、对常用检测模型做压缩
- 3、部署（1、针对4bits/三值/二值等的量化卷积；2、终端DL框架（如MNN，NCNN，TensorRT等））
