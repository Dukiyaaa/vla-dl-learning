import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


# 按论文挨个搭建transformer的结构
# ============== 输入部分：词嵌入向量 + 位置编码 ===============
# 词嵌入向量:输入和输出端都会用到
class Embeddings(nn.Module):
    def __init__(self, v, d_model):
        super(Embeddings, self).__init__()
        # 两个重要参数，一个是词表大小v，一个是词嵌入矩阵大小d_model，一般是512，其形状为v x d_model
        # nn.Embedding(v, d_model)就是声明了一个vxd_model的矩阵，这个矩阵可训练，可查表。用普通矩阵也能做，但需要手动索引，相当于用nn内置的方法更方便的实现了
        # 之后对lut输入词ID便会返回其对应的词嵌入向量
        self.lut = nn.Embedding(v, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 输出乘以√d_model：这是 Transformer 论文里的做法，使嵌入向量的量级与后续残差/位置编码在同一尺度，稳定训练、加快收敛
        return self.lut(x) * math.sqrt(self.d_model)


# 位置编码：关键是PE相关的两个公式
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.05)
        # 如果词嵌入向量是横向的一行向量的话，对应的位置编码向量也是同样的
        # 位置编码的值是固定的，只跟pos(词的位置)和i(列数有关)
        # 由于词的维度(v)不确定，所以用一个最大值max_len=5000来进行覆盖，建立一个max_len x d_model的全零矩阵
        # 举个例子，如果输入的词个数是20，那么就只用看前20个pos的位置编码向量，剩余的4800行不用管
        PE = torch.zeros(max_len, d_model, device="cpu")

        # torch.arange创建一个0-max_len-1的一位行向量(内容为浮点数)，unsqueeze(dim=1)则是在第1个维度(列方向，行方向是第0个维度)上增加一个维度，把行向量变成列向量
        # 变成列向量的原因是后面会有position * div的步骤，需要满足向量相乘的格式要求：列向量*行向量等于定值
        position = torch.arange(0., max_len, device="cpu").unsqueeze(dim=1)

        # 下面开始定义分母 10000^(2i/d_model)
        # 由于直接表示的话，i过大可能导致数值爆炸，故用e和ln进行转化
        # 分母 = (10000^(2i/d_model))^(-1) = e^(ln(10000(-2i/d))) = e^(2i*(-ln(10000)/d))，得到一个行向量
        div = torch.exp(torch.arange(0., d_model, 2, device="cpu") * -(math.log(10000) / d_model))

        # 根据PE公式得到奇偶pos的位置编码
        # PE[:, 0::2]表示，从行维度(0)上取所有，列维度上从第 0 列开始，每隔 2 列取一列
        PE[:, 0::2] = torch.sin(position * div)
        # PE[:, 1::2]表示，从行维度(0)上取所有，列维度上从第 1 列开始，每隔 2 列取一列
        PE[:, 1::2] = torch.cos(position * div)

        # 由于后续训练时，数据是以batch传入，也就是会有多个词嵌入矩阵，因此给PE再加一个维度以匹配batch
        PE = PE.unsqueeze(dim=0)
        # 将PE矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('PE', PE)

    def forward(self, x):
        # x是传入的一个batch的所有词嵌入矩阵，形状为 [batch_size, seq_len, d_model]
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        # PE[:, :x.size(1), :]表示，第一个维度(batch维)取所有，第二个维度取词数量个(最大为max_len个)，第三个维度取所有(d_model个)
        x = x + self.PE[:, :x.size(1), :]

        # 防过拟合
        return self.dropout(x)


# ============== 注意力机制：(qk + softmax)v ===============
def attention(query, key, value, mask=None, dropout=None):
    # q,k,v三个矩阵作为参数传入，后续可训练
    # 第一个公式，a = q * k，注意k要转置才能相乘。最后还要除以根号dk，dk由人为规定，是qkv矩阵的列参数
    d_k = query.size(-1)
    # transpose(-2, -1) 交换最后两个维度，也就是把 [batch, seq_len, d_k] 变成 [batch, d_k, seq_len]；改成(-1,-2)也行
    a = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 根据传入参数确定是否给一半矩阵加上掩码
    if mask is not None:
        # 在 a 上操作，将 mask==0 的位置设为 -1e9
        a = a.masked_fill(mask == 0, -1e9)

    # 接softmax，-1维度才是横向维度，代表每个词对其余词的关注度
    A = F.softmax(a, dim=-1)

    # dropout
    if dropout is not None:
        A = dropout(A)

    # 返回加权求和，以及得分矩阵A
    return torch.matmul(A, value), A


# ============== 多头注意力机制：把d_model分给多个头做并行处理 ===============
def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 断言语句保证h能整除d_model,如果不能，程序就会终止
        assert d_model // h == 0
        # 单个头的qkv矩阵维度
        self.d_k = d_model // h
        self.h = h

        # 输入乘三个w矩阵(形状为d x dk)可以获得qkv三个矩阵，这个过程可以用全连接层来表示，其本身就支持训练
        # 注意参数是d_model,因为多头注意力中，输入的d_model被分成了h份，同时qkv也是在 被计算出来后 被分成了h份
        # nn.Linear 对输入张量的最后一维做线性变换,所以在声明Linear参数时只关心d_model
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # linears[0] → WQ：把输入 X 投影成 Query
        # linears[1] → WK：把输入 X 投影成 Key
        # linears[2] → WV：把输入 X 投影成 Value
        # linears[3] → WO：把多头拼接结果投影成最终输出
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 取batch size
        nbatches = query.size(0)

        # 假设输入 query, key, value 都是 [batch, seq_len, d_model]
        # 1. 先分别做线性变换（投影）
        Q = self.linears[0](query)  # [batch, seq_len, d_model]
        K = self.linears[1](key)  # [batch, seq_len, d_model]
        V = self.linears[2](value)  # [batch, seq_len, d_model]

        # 2. 把每个矩阵拆成 h 个头（即把 d_model 切成 h×d_k）
        # 形状变化: [batch, seq_len, d_model] → [batch, seq_len, h, d_k]
        Q = Q.view(nbatches, -1, self.h, self.d_k)
        K = K.view(nbatches, -1, self.h, self.d_k)
        V = V.view(nbatches, -1, self.h, self.d_k)

        # 3. 调换维度，把头放到第二维（seq_len 前面）
        # 形状变化: [batch, seq_len, h, d_k] → [batch, h, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # 现在 Q, K, V 的形状都是 [batch, h, seq_len, d_k]，它们可以直接送入 attention 函数

        # mask的原始形状: [batch, seq_len, seq_len]，为了匹配维度，也需要在dim=1的地方加一个维
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 将QKV送入attention函数，得到的x的形状为[batch, h, seq_len, d_k]
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)
        # 将输出连接起来
        # 第一步：交换“头数”和“序列长度”
        # [batch, h, seq_len, d_k] → [batch, seq_len, h, d_k]
        x = x.transpose(1, 2)
        # 第二步：保证内存连续（为下一步 view 做准备）
        x = x.contiguous()
        # 第三步：把所有的“头”拼接到特征维度上
        # [batch, seq_len, h, d_k] → [batch, seq_len, h * d_k] = [batch, seq_len, d_model]
        x = x.view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


# ============== 层归一化：有对应库函数实现，但手动写用于学习理解 ===============
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 公式：self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2
        # 加了一个self.eps小偏移，因为要防止分母为0(std标准差有为0的风险)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # features通道等于d_model，因为输入形状为[batches, seq_len, d_model]
        self.eps = eps

    def forward(self, x):
        # 求x的最后一个维度d_model的均值和方差
        # x的形状为[batches, seq_len, d_model]，也就是有batches个样本，每个样本有seq_len个token
        # 对于每个token，在d_model上取其均值和方差做处理，这样输出依旧为[batches, seq_len, d_model]，但每个token d_model上的数值已被归一化
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return ((x - mean) / (math.sqrt(std**2 + self.eps))) * self.a_2 + self.b_2


# ============== 前馈神经网络 ===============
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 网络组成：两个全连接层+激活函数
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.w_1(x))
        # d_ff通常更大，dropout放在这里更合适
        x = self.dropout(x)
        x = self.w_2(x)

        return x


# ============== SubLayerConnection ===============
class SublayerConnection(nn.Module):
    def __init__(self, features, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer是外部声明的方法，作为参数传进来；比如可以是一个多头注意力类/feedforward的实例
        return x + self.dropout(sublayer(self.norm(x)))


