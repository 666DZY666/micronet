__version__ = "0.1.2.r"

def quant_test():
    import torch.nn as nn
    import torch.nn.functional as F

    # ``quantize`` is quant_module, ``QuantConv2d`` and ``QuantLinear`` are quant_op
    from micronet.compression.quantization.wbwtab.quantize import QuantConv2d as quant_conv_wbwtab
    from micronet.compression.quantization.wqaq.dorefa.quantize import QuantConv2d as quant_conv_dorefa
    from micronet.compression.quantization.wqaq.dorefa.quantize import QuantLinear as quant_linear_dorefa
    from micronet.compression.quantization.wqaq.iao.quantize import QuantConv2d as quant_conv_iao
    from micronet.compression.quantization.wqaq.iao.quantize import QuantLinear as quant_linear_iao

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
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

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
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

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    class QuantLeNetIAO(nn.Module):
        def __init__(self):
            super(QuantLeNetIAO, self).__init__()
            self.conv1 = quant_conv_iao(1, 10, kernel_size=5, device='cpu')
            self.conv2 = quant_conv_iao(10, 20, kernel_size=5, device='cpu')
            self.fc1 = quant_linear_iao(320, 50, device='cpu')
            self.fc2 = quant_linear_iao(50, 10, device='cpu')

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    lenet = LeNet()
    quant_lenet_wbwtab = QuantLeNetWbWtAb()
    quant_lenet_dorefa = QuantLeNetDoReFa()
    quant_lenet_iao = QuantLeNetIAO()
    print('quant_model is ready')
    print('micronet is ready')