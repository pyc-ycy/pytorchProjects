from top import *

from MyEmbedding import MyEmbedding
from PositionalEncoding import PositionalEncoding
from utils import subsequent_mask, attention
from MutiHeadAttention import MutilHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from LayerNorm import LayerNorm
from SublayerConnection import SubLayerConnection
from EncoderLayer import EncoderLayer

def use() :
    d_model = 512
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    # 文本嵌入层
    emb = MyEmbedding(d_model, vocab)
    embr = emb(x)
    dropout = 0.1
    max_len = 60
    y = embr
    # 位置编码层
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(y)
    query = key = value = pe_result
    mask = Variable(torch.zeros(2, 4, 4))
    # 注意力机制
    attn, p_attn = attention(query=query, key=key, value=value, mask=mask)
    head = 8
    embedding_dim = 512
    dropout = 0.2
    # 多头注意力机制
    mha = MutilHeadAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    d_ff = 64
    dropout = 0.2
    x = mha_result
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(x)
    # 规范化层
    features = d_model
    eps = 1e-6
    x = ff_result
    ln = LayerNorm(features, eps)
    ln_result = ln(x)
    # 子层连接结构
    x = pe_result
    size = d_model
    sublayer = lambda x: mha(x, x, x, mask)
    sc = SubLayerConnection(size, dropout)
    sc_result = sc(x, sublayer)
    # 编码器层
    el = EncoderLayer(size, mha, ff, dropout)
    el_result = el(x, mask)
    print(el_result)
    print(el_result.shape)
    

if __name__ == "__main__":
    use()
    # x = torch.randn(4, 4)
    # print(x)
    # y = x.view(16)
    # print(y)
    # print(y.size())
    # z = x.view(-1, 8)
    # print(z.size())
    # print(z)
    # a = torch.randn(1, 2, 3, 4)
    # print(a)
    # print(a.size())
    # b = a.transpose(1,2)
    # print(b.size())
    # print(b)
    # c = a.view(1, 2, 3, 4)
    # print(c.size())
    # print(c)
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
