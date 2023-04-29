from top import *

# 构建前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model：代表词嵌入的维度，同时也是两个线性层的输入和输出维度
        # d_ff：代表第一个线性层输出维度，和第二个线性层的输入维度
        # dropout：经过 Dropout 层处理时，随机置零的比率
        super(PositionwiseFeedForward, self).__init__()

        # 定义两层全连接的线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: 代表上一层的输出
        # 首先将 x 送入第一个线性层网络，然后经过 relu 函数的激活，再经历 dropout 层处理
        return self.w2(self.dropout(F.relu(self.w1(x))))
    

