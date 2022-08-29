from random import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Loader():
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def train_val_loader(self, train_batch_size, val_batch_size, shuffle=True, split=1, ratio=1, manual = False, new_size = 0):
        if (manual == False):
            dataset_size = int(self.train.__len__() * ratio)
        else:
            dataset_size = new_size
        
        rest = int(self.train.__len__()) - dataset_size
        new_train, _ = random_split(self.train, [dataset_size, rest])

        train_size = int(new_train.__len__() * split)
        val_size = int(new_train.__len__() - train_size)
        train, val = random_split(new_train, [train_size, val_size])
        train_loader = DataLoader(train, batch_size=train_batch_size,
                                  shuffle=shuffle)
        val_loader = DataLoader(val, batch_size=val_batch_size,
                                shuffle=shuffle)

        return train_loader, val_loader

    def test_loader(self, test_batch_size, shuffle = True, ratio = 1):
            
        test_loader = DataLoader(self.test, batch_size=test_batch_size,
                                shuffle=shuffle)
        return test_loader
        
    def visualize(self): 
        pass

class cifar10loader(Loader):

    def __init__(self, mytransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
                                    download = True, root = './data'):

        self.trainset = torchvision.datasets.CIFAR10(root=root, train=True, transform=mytransform, download=download)
        self.testset = torchvision.datasets.CIFAR10(root=root, train=False, transform=mytransform, download=download)
        super().__init__(self.trainset, self.testset)
    
    def visualize(self, batch, labels):
       
        batch = batch/2 + 0.5    
                    
        fig = plt.figure(figsize=(10,7))
       
        for i in range(batch.shape[0]):
            fig.add_subplot(3,int(batch.shape[0]/3) + 1, i+1)
            image = torch.squeeze(batch[i]).numpy() 
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.axis('off')
            plt.title(labels[i].item())
       
        plt.show()

class mnistLoader(Loader):

    def __init__(self, mytransform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.7,), (0.7,))
                             ]), download = True, root = './data'):

        self.trainset = torchvision.datasets.MNIST(root=root, train=True, transform=mytransform, download=download)
        self.testset = torchvision.datasets.MNIST(root=root, train=False, transform=mytransform, download=download)
        super().__init__(self.trainset, self.testset)
    
    def visualize(self, batch, labels):
       
        batch = batch/2 + 0.5 
        print(batch.shape)                
        fig = plt.figure(figsize=(10,7))  
        for i in range(batch.shape[0]):
            fig.add_subplot(3,int(batch.shape[0]/3) + 1, i+1)
            image = torch.squeeze(batch[i]).numpy() 
            plt.imshow(image)
            plt.axis('off')
            plt.title(labels[i].item())   
        plt.show()