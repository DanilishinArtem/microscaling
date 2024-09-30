import torch.optim as optim
import torch
import torch.nn as nn
from model import Mnist_regilar, Mnist_quantized
from config import Config
from learningProcess import LearningProcess
from mx import MxSpecs


torch.manual_seed(20)


if __name__ == "__main__":
    mx_specs = MxSpecs()
    mx_specs['scale_bits'] = Config.scale_bits
    mx_specs['w_elem_format'] = Config.w_elem_format
    mx_specs['a_elem_format'] = Config.a_elem_format
    mx_specs['block_size'] = Config.block_size
    mx_specs['bfloat'] = Config.bfloat
    mx_specs['custom_cuda'] = Config.custom_cuda

    if Config.quantizied:
        model = Mnist_quantized(mx_specs)
    else:
        model = Mnist_regilar(mx_specs)

    # optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    learner = LearningProcess(optimizer, criterion)
    if Config.quantizied:
        print('Results for quantized model')
    else:
        print('Results for regular model')
    learner.train(model, draw=True)