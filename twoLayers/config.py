

class Config:
    epochs = 5000
    lr = 0.01
    quantizied = False
    scale_bits = 8
    w_elem_format = 'fp8_e5m2'
    a_elem_format = 'fp8_e5m2'
    block_size = 32
    bfloat = 16
    custom_cuda = True