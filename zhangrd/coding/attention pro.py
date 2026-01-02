import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional,Type

class MultiheadAttention(nn.Module):
    def __init__(
            self,
            hidden_dim:int,  #输出，输入特征维度（必须能被num_heads整除）
            num_heads:int, #注意力头的数量
            attn_drop:float=0.0,  #注意力权重的droppout概率
            proj_drop:float=0.0,  #注意力权重的droppout概率
            out_attention:bool=False, #是否返回注意力权重
            qkv_bias:bool=False, #Q,K,V线性层是否使用偏置
            qk_norm:bool=False, #是否对每个头Q和K进行归一化
            scale_norm:bool=False, #是否在输出前对整个特征做归一化
            norm_layer:Optional[Type[nn.Module]]=None, #归一化层类型（如nn.Layernorm）
            device=None, #设备（如：cuda）
            detype=None, #数据类型(如float.32)
              )-> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert hidden_dim % num_heads == 0 #确保hidden_dim能被num_heads整除
        self.out_attention = out_attention
        dd={"device":device,"detype":detype} #传递设备和数据类型
        self.qk_norm=qk_norm
        #如果启用了qk_norm或scale_norm，必须提供norm_layer;
        if self.qk_norm or scale_norm:
            assert(
                norm_layer is None
            )
        #为Q,K,V创建独立的线性投影层
        self.W_q = nn.Linear(self.hidden_dim,self.hidden_dim,bias=qkv_bias)
        self.W_k = nn.Linear(self.hidden_dim,self.hidden_dim,bias=qkv_bias)
        self.W_v = nn.Linear(self.hidden_dim,self.hidden_dim,bias=qkv_bias)

        #如果启用qk_norm，则对每个头Q和K做归一化(按head_dim)
        self.q_norm = norm_layer(self.head_dim,**dd) if self.qk_norm else nn.Identity()
        self.k_norm = norm_layer (self.head_dim,**dd) if self.qk_norm else nn.Identity()

        # 如果启用 scale_norm，则在拼接后对整个 hidden_dim 做归一化
        self.norm = norm_layer(self.hidden_dimi,**dd) if scale_norm else nn.Identity()

        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor, #输入张量，形状为（B,N,C），B=batch size,N=序列长度（或token数量），C=hidden_dim 特征维度
            mask: Optional[torch.Tensor]=None,#mask:可选注意力掩码，形状与x一致，（但逻辑上应该广播到（B，nh，N,N）），实际使用中通常为 (B, 1, 1, N) 或 (B, N, N)。此处实现假设 mask 已适配
        )-> torch.Tensor:
        B,N,C=x.shape
        qkv=[]
        for matrix in [self.W_q,self.W_k,self.W_v]:
            temp=matrix(x) #（B,N,C）
            #重排多头格式（B,num_heads,N,head_dim）
            temp=rearrange(temp,
                           pattern="b n (nh nd) -> b n nh nd",
                           nh=self.num_heads,
                           nd=self.head_dim
            )
            qkv.append(temp)
        q,k,v=qkv #每个都是（B，nh，N，nd）

        # 可选：对Q和K按head_dim归一化，常用于稳定训练
        if self.qk_norm:
            q=self.q_norm(q)
            k=self.k_norm(k)

        # 计算缩放点积注意力分数
        # q @ k^T 得到 (B, nh, N, N)，再乘以 1/sqrt(head_dim) 缩放
        attn_scores = q @ k.transpose(-1, -2) * (self.head_dim ** (-0.5))  # (b,nh,n,n)

        if mask is not None:
            # 应用注意力掩码（注意：通常 mask 是加在 softmax 前的负无穷大值）
            assert mask.shape == x.shape
            attn_scores +=mask

        # 对最后一个维度（即 key 维度）做 softmax，得到注意力权重
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        # 加权求和：attn @ V
        y = attn_scores @ v  # (B, nh, N, nd)

        # 将多头结果拼接回 (B, N, C)
        y = rearrange(
            y, pattern="b nh n nd->b n (nh nd)", nh=self.num_heads, nd=self.head_dim
        )
        y = self.norm(y)
        y = self.out_proj(y)
        y = self.proj_drop(y)

        if self.out_attention:
            return y, attn_scores
        else:
            return y

if __name__ == "__main__":
    x = torch.randn((128, 4, 256))
    mha = MultiheadAttention(256, 4, 0.1, out_attention=False)

    y = mha(x)

    print(y.shape)


