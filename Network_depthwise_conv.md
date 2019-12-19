# 原始的卷积

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): FP_Conv2d(
      (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (6): FP_Conv2d(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): FP_Conv2d(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (8): AvgPool2d(kernel_size=3, stride=2, padding=1)
    (9): FP_Conv2d(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (10): FP_Conv2d(
      (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (11): Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```



# 原始的深度可分离卷积

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv1): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), groups=32)
      (conv2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): FP_Conv2d(
      (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), groups=64)
      (conv2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
      (conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (6): FP_Conv2d(
      (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256)
      (conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): FP_Conv2d(
      (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), groups=256)
      (conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): FP_Conv2d(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (10): FP_Conv2d(
      (conv1): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), groups=512)
      (conv2): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (11): Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```



# 深度可分离卷积+对应的1*1卷积通道倍增

```c++
Net(
  (tnn_bin): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): FP_Conv2d(
      (conv1): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), groups=32)
      (conv2): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): FP_Conv2d(
      (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), groups=64)
      (conv2): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): FP_Conv2d(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
      (conv2): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (6): FP_Conv2d(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), groups=256)
      (conv2): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (7): FP_Conv2d(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), groups=256)
      (conv2): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): FP_Conv2d(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (conv2): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (10): FP_Conv2d(
      (conv1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), groups=512)
      (conv2): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1))
      (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (11): Conv2d(2048, 10, kernel_size=(1, 1), stride=(1, 1))
    (12): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): AvgPool2d(kernel_size=8, stride=1, padding=0)
  )
)
```

