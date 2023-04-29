from top import *

# 规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features 代表词嵌入的维度
        # eps：一个足够小的正数，用来在规范化计算公式的分母中，防止除零操作
        super(LayerNorm, self).__init__()

        # 初始化两个参数张量 a2，b2，用于对结果做规范化操作计算
        # 将其用 nn.Parameter 进行封装，代表他们也是模型中的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x 代表上一网络层的输出
        # 首先对 x 进行最后一个维度的求均值操作，同时保持输出维度和输入维度一致
        mean = x.mean(-1, keepdim=True)
        # 接着对 x 进行最后一个维度上的求标准差操作，同时保持输出维度和输入维度一致
        std = x.std(-1, keepdim=True)
        # 按照规范化公式进行计算
        return self.a2 * (x - mean) / (std + self.eps) + self.b2
