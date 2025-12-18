from turtle import forward
from xml.sax.handler import version
import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, out_attention, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_attention = out_attention

        self.q_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.k_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.v_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D = x.shape

        q, k, v = (
            self.q_matrix(x),
            self.q_matrix(x),
            self.q_matrix(x),
        )  # (b,n,d)

        #  计算注意力分数

        attn_scores = q @ k.transpose(-1, -2) * (self.hidden_dim ** (-0.5))  # (b,n,n)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        #
        y = attn_scores @ v  # (b,n,d)

        if self.out_attention:
            return y, attn_scores
        else:
            return y


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout, out_attention, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim
        self.out_attention = out_attention

        self.q_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.k_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.v_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, mask=None):
        B, C, D = x.shape

        q, k, v = (
            self.q_matrix(x),
            self.q_matrix(x),
            self.q_matrix(x),
        )  # (b,n,d)

        q = rearrange(
            q, pattern="'b n (nh nd)'->'b nh n nd'", nh=self.num_heads, nd=self.head_dim
        )
        k = rearrange(
            k, pattern="'b n (nh nd)'->'b nh n nd'", nh=self.num_heads, nd=self.head_dim
        )
        v = rearrange(
            v, pattern="'b n (nh nd)'->'b nh n nd'", nh=self.num_heads, nd=self.head_dim
        )
        #  计算注意力分数

        attn_scores = q @ k.transpose(-1, -2) * (self.head_dim ** (-0.5))  # (b,nh,n,n)

        if mask is not None:
            assert mask.shape == x.shape
            attn_scores += mask

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        #
        y = attn_scores @ v  # (b,nh,n,nd)

        y = rearrange(
            y, pattern="'b nh n nd'->'b n (nh nd)'", nh=self.num_heads, nd=self.head_dim
        )

        y = self.out_proj(y)

        if self.out_attention:
            return y, attn_scores
        else:
            return y
