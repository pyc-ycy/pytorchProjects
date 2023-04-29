import torch
import numpy as np

if __name__ == "__main__":
    # 张量
    a = torch.randn(2, 3)
    print(a.type())
    print(type(a))
    print(isinstance(a, torch.FloatTensor))
    print(isinstance(a, torch.cuda.FloatTensor))
    b = a.cuda()
    print(isinstance(b, torch.cuda.FloatTensor))
    print(isinstance(b, torch.FloatTensor))
    # 标量 dim=0
    print(torch.tensor(1.))
    print(torch.tensor(1.3))
    c = torch.tensor(2.2)
    print(c.shape)
    print(len(c.shape))
    print(c.size())
    # dim = 1，维度为1
    print(torch.tensor([1.1]))
    print(torch.tensor([1.1, 2.2]))
    print(torch.tensor(1))
    print(torch.FloatTensor(2))
    data = np.ones(2)
    print(torch.from_numpy(data))
    print(torch.ones(2).shape)
    # dim=1
    a = torch.randn(2, 3)
    print(a)
    print(a.shape)
    print(a.size(0))
    print(a.size(1))
    print(a.shape[1])
    # dim = 3
    b = torch.rand([1, 2, 3])
    print(b)
    print(b.shape)
    print(b[0])
    print(list(b.shape))
    print(b.numel())
    print(b.dim())
    # dim = 4
    print(torch.rand(2, 3, 28, 28))
    print(torch.rand(2, 3, 28, 28).shape)
    # 查看数据所占内存大小
    print(torch.rand(2, 3, 28, 28).numel())
    print(torch.rand(2, 3, 28, 28).dim())
    

    

