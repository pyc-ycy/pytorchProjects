from top import *

def subsequent_mask(size):
    # size 代表掩码张量后两个维度，形成一个方阵
    attn_shape = (1, size, size)

    # 使用 np.ones() 先构建一个全 1 的张量，然后利用 np.triu() 形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #使得三角矩阵进行反转
    return torch.from_numpy(1 - subsequent_mask)

def attention(query, key, value, mask=None, dropout=None):
    # query, key, value: 代表注意力的三个输入张量
    # mask：掩码张量
    # dropout：传入的 Dropout 实例化对象
    # 首先将 query 的最后一个维度提取出来，代表词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力计算公式，将 query 和 key 的转置矩阵进行矩阵乘法，然后除以缩放系数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 判断是否使用掩码张量
    if mask is not None:
        # 利用 masked_fill 方法，将掩码张量和 0 进行比较，若等于0，则替换成一个非常小数值
        scores = scores.masked_fill(mask==0, -1e9)
    
    # 对 scores 的最后一个维度进行 softmax 操作
    p_pattn = F.softmax(scores, dim=-1)

    # 判断是否使用 dropout
    if dropout is not None:
        p_pattn = dropout(p_pattn)
    
    # 最后一步完成 p_attn 和 value 张量的乘法，并返回 query 的注意力表示
    return torch.matmul(p_pattn, value), p_pattn

# 实现克隆函数，用于克隆线性变化层，因此需要使用 clone 函数将他们一同初始化到网络层对象列表中
def clones(moudle, N):
    # model: 代表要克隆的目标网络层
    # N：将 model 克隆几个
    return nn.ModuleList([copy.deepcopy(moudle) for _ in range(N)])
