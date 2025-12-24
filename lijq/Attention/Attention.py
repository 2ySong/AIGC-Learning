import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange # einops便捷张量操作
from typing import Optional, Type
# Optional 告诉调用者“可能返回 None”；Type 告诉调用者“请把类本身传进来”。
# 两者都不产生运行时开销，纯做“静态提示”。

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int, # 隐藏层维度，=queries/keys/values size
        num_heads: int =8,
        bias: bool =False,
        drop_rate: float =0,
    ) -> None: # 不返回实例
        super().__init__() # 必须调用父类的初始化函数
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        # 确保head_dim为整除结果
        assert self.head_dim * num_heads == hidden_dim

        # 定义线性层
        self.W_q =nn.Linear(self.hidden_dim, self.hidden_dim, bias =bias)
        self.W_k =nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.W_v =nn.Linear(self.hidden_dim, self.hidden_dim, bias =bias)
        # output layer
        self.W_o =nn.Linear(self.hidden_dim, self.hidden_dim, bias =bias)

        self.att_dropout =nn.Dropout(drop_rate)
        self.out_dropout =nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor]= None,
    ) -> torch.Tensor:
        B, L, D = x.shape # (batch_size, length, hidden_dim)

        # output dim: (batch_size, num_heads, length, head_dim)
        q = rearrange(self.W_q(x), 'b l (nh h) -> b nh l h', nh=self.num_heads)
        k  = rearrange(self.W_k(x), 'b l (nh h) -> b nh l h', nh=self.num_heads)
        v = rearrange(self.W_v(x), 'b l (nh h) -> b nh l h', nh=self.num_heads)

        # calculate dot-product attention scores Q @ K^T / sqrt(head_dim)
        # @运算符 (即 torch.matmul)支持高维张量（≥3维），自动识别最后两维进行矩阵乘
        # output dim:(batch_size, num_heads, length, length)
        att_scores = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)
        att_scores = F.softmax(att_scores, dim=-1)
        att_scores =self.att_dropout(att_scores)

        y = att_scores @ v # output dim: (batch_size, num_heads, length, head_dim)
        y = rearrange(y, 'b nh l h -> b l (nh h)') # # output dim: (batch_size, length, num_head*head_dim =hidden_dim)
        output =self.W_o(y)
        output =self.out_dropout(output)

        return output

if __name__=='__main__':
    pass
    attention =MultiHeadAttention(256, 8, False, 0.1)
    print(attention.eval())

    x = torch.randn((128, 3, 256))
    output =attention(x)
    print(output.shape)

