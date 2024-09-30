
class Config:
        batch_size = 64
        learning_rate = 0.01
        num_epochs = 3
        split = 1
        pathToData = "./mnist_dataset"

        quantizied = True
        scale_bits = 8
        w_elem_format = 'fp8_e5m2'
        a_elem_format = 'fp8_e5m2'
        block_size = 32
        bfloat = 16
        custom_cuda = True