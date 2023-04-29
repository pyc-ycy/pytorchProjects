import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

from MyEmbedding import MyEmbedding
from PositionalEncoding import PositionalEncoding

if __name__ == "__main__":
    # embedding = nn.Embedding(10, 3)
    # input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    # print(embedding(input1))

    # embedding = nn.Embedding(10, 3, padding_idx=0)
    # input1 = torch.LongTensor([0, 2, 0, 5])
    # print(embedding(input1))

    d_model = 512
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    emb = MyEmbedding(d_model, vocab)
    embr = emb(x)
    dropout = 0.1
    max_len = 60
    y = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(y)
    print(pe_result)
    print(pe_result.shape)


    # m = nn.Dropout(p=0.2)
    # input1 = torch.randn(4, 5)
    # output1 = m(input1)
    # print(output1)

    # x = torch.tensor([1, 2, 3, 4])
    # print(torch.unsqueeze(x,1))
    # print(torch.unsqueeze(x, 1))
