import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 eps=1e-5,
                 momentum=0.1):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [192, 160, 96, 192, 192, 192, 192, 192]
        self.model = nn.Sequential(
            ConvBNReLU(3, cfg[0], kernel_size=5, stride=1, padding=2),
            ConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0),
            ConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ConvBNReLU(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2),
            ConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0),
            ConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1),
            ConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0),
            ConvBNReLU(cfg[7], 10, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x
