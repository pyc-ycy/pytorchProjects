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
from utils import subsequent_mask, attention

def use() :
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
    query = key = value = pe_result
    mask = Variable(torch.zeros(2, 4, 4))
    attn, p_attn = attention(query=query, key=key, value=value, mask=mask)
    

if __name__ == "__main__":
    use()
    # sm = subsequent_mask(5)
    # print("sm: ", sm)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(sm[0])
    # plt.show()
    # 利用三角矩阵进行掩码
    # matrix = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    # tri_matrix = np.triu(matrix, k=-1)
    # print(tri_matrix)
    # print(np.triu(matrix, 0))

    # embedding = nn.Embedding(10, 3)
    # input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    # print(embedding(input1))

    # embedding = nn.Embedding(10, 3, padding_idx=0)
    # input1 = torch.LongTensor([0, 2, 0, 5])
    # print(embedding(input1))

    # # 设置画布
    # plt.figure(figsize=(15, 5))

    # # 实例化 PositionalEncoding 类对象，词嵌入维度 20，置零比例未 0
    # pe2 = PositionalEncoding(20, 0)

    # t = pe2(Variable(torch.zeros(1, 100, 20)))
    # plt.plot(np.arange(100), t[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])
    # plt.show()
    


    # m = nn.Dropout(p=0.2)
    # input1 = torch.randn(4, 5)
    # output1 = m(input1)
    # print(output1)

    # x = torch.tensor([1, 2, 3, 4])
    # print(torch.unsqueeze(x,1))
    # print(torch.unsqueeze(x, 1))

    # x = Variable(torch.randn(5, 5))
    # mask = Variable(torch.zeros(5, 5))
    # y = x.masked_fill(mask==0, -1e9)
    # print(y)
