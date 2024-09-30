from torch.utils.data import DataLoader, random_split
from dataLoader import load_mnist_dataset
from config import Config
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt


class LearningProcess:
    def __init__(self, optimizer: optim, criterion: nn.Module):
        self.train_loader = self.createDataset()
        self.optimizer = optimizer
        self.criterion = criterion

    def createDataset(self):
        dataset = load_mnist_dataset()
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_size = int(Config.split * len(train_dataset))

        train_dataset, val_dataset = random_split(train_dataset, [train_size, 0])
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        return train_loader
    
    def train(self, model: nn.Module, draw: bool = False):
        print("start training\n\n")
        total_counter = 0
        total_loss = 0
        correct = 0
        numPic = 0
        mean_time = []
        losses = []
        for epoch in range(Config.num_epochs):
            model.train()
            model = model.to('cuda')
            for batch in self.train_loader:
                start = time.time()
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                images = images.to('cuda')
                labels = labels.to('cuda')
                self.optimizer.zero_grad()

                output = model(images)
                loss = self.criterion(output, labels)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                numPic += len(images)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                dt = time.time() - start
                mean_time.append(dt)
                # losses.append(total_loss / total_counter)
                losses.append(loss.item())
                print(f"For step {total_counter}: training loss = {round(total_loss / total_counter,2)}, training accuracy = {round(correct / numPic,2)}, time per iteration = {dt}")
        
        print('Mean time of learning: {}'.format(np.mean(mean_time)))
        if draw:
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            if Config.quantizied:
                plt.savefig('./quantizied_loss_' + str(Config.num_epochs) + 'epochs.png')
            else:
                plt.savefig('./regular_loss_' + str(Config.num_epochs) + 'epochs.png')
