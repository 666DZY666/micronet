# 网络结构细节对比

## 原始网络结构

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(192, 160, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(160, 96, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(96, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (8): AvgPool2d(kernel_size=3, stride=2, padding=1)
    (9): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (11): Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```



## 采用分组卷积结构(和原始网络一致

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(192, 160, kernel_size=(1, 1), stride=(1, 1), groups=2)
      (bn): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(160, 96, kernel_size=(1, 1), stride=(1, 1), groups=2)
      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), groups=4)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), groups=4)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), groups=8)
      (bn): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (11): Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```

## 采用分组卷积结构(和原始网络不一致)

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=2)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=2)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=4)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=4)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): FP_Conv2d(
      (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), groups=8)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (11): Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```



## 原模型常规剪枝(比例为0.5)

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 47, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(47, 93, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(93, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(93, 47, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(47, 122, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (bn): BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(122, 116, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(116, 90, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(90, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (8): AvgPool2d(kernel_size=3, stride=2, padding=1)
    (9): FP_Conv2d(
      (conv): Conv2d(90, 105, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn): BatchNorm2d(105, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(105, 83, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(83, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (11): Conv2d(83, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```



## 原模型常规剪枝(比例为0.5)+规整剪枝(通道数为8)

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(48, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (bn): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(120, 88, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (8): AvgPool2d(kernel_size=3, stride=2, padding=1)
    (9): FP_Conv2d(
      (conv): Conv2d(88, 104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn): BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(104, 80, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (11): Conv2d(80, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```

