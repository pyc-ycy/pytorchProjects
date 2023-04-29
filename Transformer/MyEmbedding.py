import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 构建 Embedding 类实现文本嵌入层
class MyEmbedding(nn.Module):

    def __init__(self, d_model, vocab):
        # d_model: 词嵌入的维度
        # vocab：词表的大小
        super(MyEmbedding, self).__init__()
        # 定义 Embedding 层
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model

    def forward(self, x):
        # x: 代表输入模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)