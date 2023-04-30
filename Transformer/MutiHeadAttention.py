from top import *
from utils import *



# 实现多头注意力机制
class MutilHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head：代表几个头
        # embedding_dim: 代表词嵌入的维度
        # dropout：进行 Dropout 操作时，置零比例
        super(MutilHeadAttention, self).__init__()

        # 确认：多头的数量 head 需要整除词嵌入维度 embedding_dim
        assert embedding_dim % head == 0

        # 得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        # 获得线性层，获得 4 个，分别是 Q，K，V 以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim,embedding_dim), 4)

        # 初始化注意力张量
        self.atta = None

        # 初始化 dropout 对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query，key，value 表示注意力机制的三个输入张量，mask 表示掩码张量
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 使用 unsqueeze 将掩码张量进行维度扩充，代表多头的第 n 个头
            mask = mask.unsqueeze(1)
        
        # 得到 batch_size
        batch_size = query.size(0)

        # 首先使用 zip 将网络层和输入数据链接在一起，模型的输出利用 view 和 transpose 进行维度和形状的改变
        query,key,value =\
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) 
         for model,x in zip(self.linears, (query,key,value))]
        
        # 将每个头的输出传入到注意力层
        x, self.atta = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到每个头的计算结果是 4 维张量，需要进行形状的转换
        # 前面已经将 1，2 两个维度进行过转置，这里从小转置回来
        # 注意：经历了 transpose（） 方法后，必须使用 contiguous 方法，不然无法使用 view() 方法
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)
        # 最后将x输入线性层列表中的最后一个线性层中进行处理，得到最终的多头注意力结果输出
        return self.linears[-1](x)
    

