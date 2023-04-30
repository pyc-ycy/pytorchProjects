from top import *
from LayerNorm import LayerNorm

# 子层连接结构
class  SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size：词嵌入维度
        super(SubLayerConnection, self).__init__()
        #实例化一个规范化层对象
        self.norm = LayerNorm(size)
        # 示例化一个 dropout 对象
        self.dropout = nn.Dropout(p=dropout)
        self.size = size
    
    def forward(self, x, sublayer):
        # x 代表上一层传入的张量
        # sublayer：该子层连接中子层函数
        # 首先将 x 进行规范化，然后送入子层函数中处理，处理结果进入 dropout 层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))

