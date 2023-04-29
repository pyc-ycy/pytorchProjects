import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 构建位置编码器
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        # d_model: 代表词嵌入的维度
        # dropout：代表 dropout 层的置零比例
        # max_len：代表每个句子的最大长度
        super(PositionalEncoding, self).__init__()

        # 实例化 dropout 层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，大小时 max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变化矩阵 div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -1 * (math.log(10000.0) / d_model))

        #将前面定义的变化矩阵进行一个奇数位置和偶数位置的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量扩张成三维张量
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册成模型的 buffer，这个 buffer 不是模型中的参数，不跟随优化器同步更新
        # 注册成 buffer 后，就可以在模型保存后重新加载的时候，将这个位置编码器和模型参数一同加载
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: 代表文本序列的词嵌入表示
        # 首先明确 pe 的编码太长了，将第二个维度，即 max_len 对应的维度，缩小成 x 的句子长度同等的长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
