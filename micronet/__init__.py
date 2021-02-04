__version__ = "1.2.0"

def quant_test_manual():
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

def quant_test_auto():
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
