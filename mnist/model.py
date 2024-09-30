from torch import nn
import torch.nn.functional as F
from mx import Conv2d, relu, adaptive_avg_pool2d, Linear, softmax
from mx import simd_split, simd_add


class Mnist_regilar(nn.Module):
    def __init__(self, mx_specs=None):
        super(Mnist_regilar, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.lin2 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu3(self.lin1(x))
        x = self.lin2(x)
        return self.log_softmax(x)
    



class Mnist_quantized(nn.Module):
    def __init__(self, mx_specs=None):
        super(Mnist_quantized, self).__init__()
        self.mx_specs = mx_specs

        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = Linear(64 * 7 * 7, 128, mx_specs)
        self.lin2 = Linear(128, 10, mx_specs)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        input = self.conv1(x)
        input = relu(input, self.mx_specs)
        input = self.pool(input)
        input = self.conv2(input)
        input = relu(input, self.mx_specs)
        input = self.pool(input)
        input = self.flatten(input)
        input = self.lin1(input)
        input = relu(input, self.mx_specs)
        input = self.lin2(input)
        return self.log_softmax(input)