import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        # dim_q,dim_k,dim_v分别是query的维度，key和query的维度，value的映射维度
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        # 权重矩阵
        self.W_q = nn.Linear(dim_q, dim_k, bias=False)
        self.W_k = nn.Linear(dim_q, dim_k, bias=False)
        self.W_v = nn.Linear(dim_q, dim_v, bias=False)
        # 归一化因子
        self._norm_fact = 1 / (dim_k ** 0.5)

    def forward(self, x):
        # batch,n,dim_q=x.shape, batch是批次大小，n是序列长度，dim_q是输入的特征维度
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact
        att = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(att, v)
        out = torch.matmul(att, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads):
        super().__init__()
        assert dim_k % num_heads == 0
        assert dim_v % num_heads == 0

        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim_k = dim_k // num_heads
        self.head_dim_v = dim_v // num_heads
        # 投影层
        self.W_q = nn.Linear(dim_in, dim_k, bias=False)
        self.W_k = nn.Linear(dim_in, dim_k, bias=False)
        self.W_v = nn.Linear(dim_in, dim_v, bias=False)

        self._norm_fact = 1 / (self.head_dim_k ** 0.5)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # (B, N, dim_in)

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 分割成多个头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact  # (B, H, N, N)
        att = F.softmax(scores, dim=-1)  # (B, H, N, N)
        head_out = torch.matmul(att, v)  # (B, H, N, head_dim_v)

        # 合并多头：(B, H, N, head_dim_v) -> (B, N, H, head_dim_v) -> (B, N, dim_v)
        out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_v)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 6, 256)
    att = SingleAttention(dim_q=256, dim_k=8, dim_v=3)
    out = att.forward(x)
    print("SingleAttention output shape:", out.shape)
    att = MultiHeadAttention(dim_in=256, dim_k=8, dim_v=4, num_heads=2)
    out = att.forward(x)
    print("MultiHeadAttention output shape:", out.shape)