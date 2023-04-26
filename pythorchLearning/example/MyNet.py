import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # xw + b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
    
    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1 + b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2 + b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3 + b3
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    

