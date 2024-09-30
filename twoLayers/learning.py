import matplotlib.pyplot as plt
from config import Config
import time
import numpy as np

def learningProcess(epochs, model, criterion, optimizer, x, y, draw: bool=False):
    model = model.cuda()
    x = x.cuda()
    y = y.cuda()
    losses = []
    mean_time = []
    for epoch in range(epochs):
        start = time.time()
        model.train()
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            dt = time.time() - start
            mean_time.append(dt)
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, time per 50 epochs: {dt}')
    print('Mean time of learning: {}'.format(np.mean(mean_time)))
    if draw:
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if Config.quantizied:
            plt.savefig('./quantizied_loss_' + str(Config.epochs) + 'epochs.png')
        else:
            plt.savefig('./regular_loss_' + str(Config.epochs) + 'epochs.png')