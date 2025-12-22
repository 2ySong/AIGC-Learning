import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()  # 必须调用父类的初始化函数
        self.dim = dim
        self.num_head = num_head
        # 头维度必须是整数，且dim必须能被num_head整除（否则需要调整dim或num_head）
        self.head_dim = dim // num_head
        # 确保dim能被num_head整除，避免维度错误
        assert self.head_dim * num_head == dim, f"dim {dim} must be divisible by num_head {num_head}"

        # 定义Q/K/V的线性层（输入输出维度都是dim）
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x的形状：(B, N, C)，其中B=批次大小，N=序列长度，C=特征维度（dim）
        B, N, C = x.shape

        # 1. 线性变换得到Q/K/V，形状：(B, N, C)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 2. 拆分多头：将C维度拆分为num_head × head_dim，并调整维度顺序
        # 步骤：(B, N, C) → (B, N, num_head, head_dim) → (B, num_head, N, head_dim)
        q = q.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数：Q @ K^T / sqrt(head_dim)
        # Q@K^T的形状：(B, num_head, N, N)
        attn_scores = q @ k.transpose(-2, -1) / sqrt(self.head_dim)
        # 4. softmax归一化（在最后一维，即每个token对其他token的注意力权重）
        attn = F.softmax(attn_scores, dim=-1)

        # 5. 注意力加权求和：attn @ V，形状：(B, num_head, N, head_dim)
        output = attn @ v

        # 6. 拼接多头：恢复维度为(B, N, C)
        # 步骤：(B, num_head, N, head_dim) → (B, N, num_head, head_dim) → (B, N, C)
        output = output.transpose(1, 2).reshape(B, N, C)

        # 7. 输出投影
        output = self.proj(output)

        return output

class SelfAttention(nn.Module):
    # dim_in: int
    # dim_k: int
    # dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        # dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # dist = torch.softmax(dist, dim=-1)  # batch, n, n
        dist2 = q @ k.transpose(1, 2) * self._norm_fact
        dist2 = torch.softmax(dist2, dim=-1)
        att = dist2 @ v

        # att = torch.bmm(dist, v)
        return att

def dropout(self,x,p=0.5):
    if not self.training:
        return x.copy()
    mask=np.random.binomial(1,p=1-p,size=x.shape)
    return x*mask/(1-p)


class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        手动实现LayerNorm
        :param normalized_shape: 要归一化的维度（比如hidden_dim=512，或(seq_len, 512)，但通常是特征维度）
        :param eps: 防止分母为0的小常数，默认1e-5
        """
        super().__init__()
        # 将normalized_shape转为tuple（兼容int和list/tuple输入）
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps

        # 可学习的缩放参数gamma（初始化为1）和平移参数beta（初始化为0）
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状需满足：最后几个维度匹配normalized_shape
                  例如：normalized_shape=(512,)，则x可以是[batch, 512]或[batch, seq_len, 512]
        :return: 归一化后的张量，形状与x一致
        """
        # ===== 步骤1：计算均值（在normalized_shape维度上）=====
        # dim：需要计算均值/方差的维度（最后len(normalized_shape)个维度）
        dims = [-i for i in range(1, len(self.normalized_shape)+1)]  # 比如normalized_shape=(512,) → dims=[-1]
        mean = x.mean(dim=dims, keepdim=True)  # keepdim=True保持维度，方便后续广播计算

        # ===== 步骤2：计算方差 =====
        var = x.var(dim=dims, keepdim=True, unbiased=False)  # unbiased=False：使用样本方差（除以d，而非d-1），与PyTorch内置一致

        # ===== 步骤3：归一化 =====
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # ===== 步骤4：缩放和平移 =====
        # gamma和beta会自动广播到x_hat的形状（因为normalized_shape匹配最后几个维度）
        y = self.gamma * x_hat + self.beta

        return y
def softmax_torch(x, dim=-1):
    """
    PyTorch实现的Softmax（支持多维张量，数值稳定）
    :param x: PyTorch张量（如[2, 3]的批次数据）
    :param dim: 计算Softmax的维度（默认最后一维，如分类任务的num_classes维度）
    :return: Softmax后的张量
    """
    max_x = torch.max(x, dim=dim, keepdim=True)[0]  # 按指定维度取最大值，keepdim保持维度匹配在指定维度上计算时，会返回一个包含两个元素的元组 ——
    # 第一个元素是维度上的最大值张量，第二个元素是最大值对应的索引张量
    exp_x = torch.exp(x - max_x)  # 数值稳定的指数计算
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)  # 按指定维度求和
    return exp_x / sum_exp_x

if __name__=="__main__":
    x=torch.rand(2,3,256)
    net=MultiHeadAttention(dim=256, num_head=4)
    print(net(x).shape)