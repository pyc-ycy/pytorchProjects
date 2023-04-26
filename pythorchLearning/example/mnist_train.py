import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot
from MyNet import MyNet


# step1 load data
def load_dataset():
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(net, train_data, print_tag):
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    train_loss = []
    for epoch in range(3):
        for batch_idx, (x,y) in enumerate(train_data):
            # x: [b, 1, 28, 28], y: [512]
            # [b, 1, 28, 28] ==> [b, 784]
            x = x.view(x.size(0), 28*28)
            # => [b, 10]
            out = net(x)
            # [b, 10]
            y_one_hot = one_hot(y)
            # loss = mse(out, y_one_hot)
            loss = F.mse_loss(out, y_one_hot)
            optimizer.zero_grad()
            loss.backward()
            # w' = w - lr*grad
            optimizer.step()
            train_loss.append(loss.item())

            if print_tag is True:
                if batch_idx % 50 == 0:
                    print(epoch, batch_idx, loss.item())
    return train_loss

def cal_accuracy(net, test_data):
    total_correct = 0
    for x,y in test_data:
        x = x.view(x.size(0), 28*28)
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct
    total_num = len(test_data.dataset)
    acc = total_correct / total_num
    return acc

def predict(net, test_data):
    x,y = next(iter(test_data))
    out = net(x.view(x.size(0), 28*28))
    pred = out.argmax(dim=1)
    plot_image(x, pred, 'test')

if __name__ == "__main__":
    train_data, test_data = load_dataset()
    # x,y = next(iter(test_data))
    # print(x.shape, y.shape, x.min(), x.min())
    # plot_image(x, y, 'image sample')
    net = MyNet()
    train_loss = train(net=net, train_data=train_data, print_tag=True)
    plot_curve(train_loss)
    acc = cal_accuracy(net, test_data=test_data)
    print("test acc:", acc)
    predict(net=net, test_data=test_data)
