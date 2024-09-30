import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleNN_quantized, SimpleNN_regular
from config import Config
from learning import learningProcess
from mx import MxSpecs


if __name__ == '__main__':
    mx_specs = MxSpecs()
    mx_specs['scale_bits'] = Config.scale_bits
    mx_specs['w_elem_format'] = Config.w_elem_format
    mx_specs['a_elem_format'] = Config.a_elem_format
    mx_specs['block_size'] = Config.block_size
    mx_specs['bfloat'] = Config.bfloat
    mx_specs['custom_cuda'] = Config.custom_cuda

    if Config.quantizied:
        model = SimpleNN_quantized(mx_specs)
    else:
        model = SimpleNN_regular(mx_specs)

    N = 100 
    x = torch.unsqueeze(torch.linspace(-10, 10, N), dim=1)
    y = 3 * x + 2 + torch.randn(x.size()) * 2
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    if Config.quantizied:
        print('Results for quantized model')
    else:
        print('Results for regular model')
    learningProcess(Config.epochs, model, criterion, optimizer, x, y, True)
