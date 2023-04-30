from top import *
from utils import *
from SublayerConnection import SubLayerConnection

# 编码器层类
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_worward, dropout):
        # size: 代表词嵌入维度
        # self_attn：代表传入的多头自注意力子层的实例化对象
        # feed_forward：前馈全连接层的实例化对象
        super(EncoderLayer, self).__init__()
        # 将两个实例化对象和参数传入类中
        self.self_attn = self_attn
        self.feed_forward = feed_worward
        self.size = size
        # 编码器层中有两个子层连接结构，使用 clones 函数进行操作
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        # x 代表上一层传入的张量
        # mask： 代表掩码张量
        # 首先让 x 经过第一个子层连接结构，内部包含多头自注意力机制
        # 再让张量经过第二个子层连接结构，其中包括前馈全连接网络
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

