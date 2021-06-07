import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, res, shortcut):
        output = res + shortcut
        return output


# *** temp_dev ***
'''
class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, res, shortcut):
        output = torch.cat([shortcut, res], dim=self.dim)
        return output
'''
