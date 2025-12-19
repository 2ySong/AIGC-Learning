import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange
from typing import Optional, Type


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        attn_drop: float = 0,
        proj_drop: float = 0,
        out_attention: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        norm_layer: Optional[Type[nn.Module]] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim

        self.out_attention = out_attention
        dd = {"device": device, "dtype": dtype}
        self.qk_norm = qk_norm
        if self.qk_norm or scale_norm:
            assert (
                norm_layer is not None
            ), "norm_layer must be provided if qk_norm or scale_norm is True"

        self.q_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.k_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.v_matrix = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, **dd) if self.qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if self.qk_norm else nn.Identity()
        self.norm = norm_layer(self.hidden_dim, **dd) if scale_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, D = x.shape
        qkv = []
        for matrix in [self.q_matrix, self.k_matrix, self.v_matrix]:
            temp = matrix(x)
            temp = rearrange(
                temp,
                pattern="b n (nh nd)->b nh n nd",
                nh=self.num_heads,
                nd=self.head_dim,
            )
            qkv.append(temp)
        q, k, v = qkv

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_scores = q @ k.transpose(-1, -2) * (self.head_dim ** (-0.5))  # (b,nh,n,n)

        if mask is not None:
            assert mask.shape == x.shape
            attn_scores += mask

        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        #
        y = attn_scores @ v  # (b,nh,n,nd)

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
    x = torch.randn((128, 3, 256))
    mha = MultiHeadAttention(256, 8, 0.1, out_attention=False)

    y = mha(x)

    print(y.shape)
