import torch
import torch.nn as nn
from config import Config
from mx import Linear, relu
from mx import simd_split, simd_add


class SimpleNN_quantized(nn.Module):
    def __init__(self, mx_specs=None):
        super(SimpleNN_quantized, self).__init__()
        self.mx_specs = mx_specs
        self.fc1 = Linear(1, 10, mx_specs)
        self.fc2 = Linear(10, 1, mx_specs)

    def forward(self, x):
        inputs = self.fc1(x)
        inputs = relu(inputs, self.mx_specs)
        inputs = self.fc2(inputs)
        return inputs


class SimpleNN_regular(nn.Module):
    def __init__(self, mx_specs=None):
        super(SimpleNN_regular, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x