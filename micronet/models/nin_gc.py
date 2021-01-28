import torch.nn as nn

def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

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
                 momentum=0.1,
                 channel_shuffle=0,
                 shuffle_groups=1):
        super(ConvBNReLU, self).__init__()
        self.channel_shuffle_flag = channel_shuffle
        self.shuffle_groups = shuffle_groups

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.channel_shuffle_flag:
            x = channel_shuffle(x, groups=self.shuffle_groups)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, cfg = None):
        super(Net, self).__init__()
        if cfg is None:
            cfg = [256, 256, 256, 512, 512, 512, 1024, 1024]
        self.model = nn.Sequential(
            ConvBNReLU(3, cfg[0], kernel_size=5, stride=1, padding=2),
            ConvBNReLU(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=0),
            ConvBNReLU(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0, groups=2, channel_shuffle=1, shuffle_groups=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ConvBNReLU(cfg[2], cfg[3], kernel_size=3, stride=1, padding=1, groups=16, channel_shuffle=1, shuffle_groups=2),
            ConvBNReLU(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=16),
            ConvBNReLU(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0, groups=4, channel_shuffle=1, shuffle_groups=4),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ConvBNReLU(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1, groups=32, channel_shuffle=1, shuffle_groups=4),
            ConvBNReLU(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0, groups=8, channel_shuffle=1, shuffle_groups=32),
            ConvBNReLU(cfg[7], 10, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x
        