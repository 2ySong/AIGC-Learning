import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional,Type

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int, #输出，输入特征维度（必须能被num_heads整除）
            num_heads: int, #注意力头的数量
            attn_drop: float = 0.0, # 注意力权重的 dropout 概率
            proj_drop: float = 0.0, # 投影层 dropout 概率
            out_attention: bool = False,  # 是否返回注意力权重
            qkv_bias: bool = False,  # Q,K,V线性层是否使用偏置
            qk_norm: bool = False,  # 是否对每个头Q和K进行归一化
            scale_norm: bool = False,  # 是否在输出前对整个特征做归一化
            norm_layer: Optional[Type[nn.Module]] = None,  # 归一化层类型（如nn.Layernorm）
            device=None,  # 设备（如：cuda）
            dtype=None,  # 数据类型(如float.32)
             )->None:
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert hidden_dim % num_heads == 0  # 确保hidden_dim能被num_heads整除
        self.out_attention = out_attention
        dd = {"device": device, "dtype": dtype}  # 传递设备和数据类型
        self.qk_norm = qk_norm
        # 如果启用了qk_norm或scale_norm，必须提供norm_layer;
        if self.qk_norm or scale_norm:
            assert (
                    norm_layer is None
            )
        # 为Q,K,V创建独立的线性投影层
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)

        # 如果启用qk_norm，则对每个头Q和K做归一化(按head_dim)
        self.q_norm = norm_layer(self.head_dim, **dd) if self.qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if self.qk_norm else nn.Identity()

        # 如果启用 scale_norm，则在拼接后对整个 hidden_dim 做归一化
        self.norm = norm_layer(self.hidden_dimi, **dd) if scale_norm else nn.Identity()

        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,  # 输入张量，形状为（B,N,C），B=batch size,N=序列长度（或token数量），C=hidden_dim 特征维度
            kv_x : Optional[torch.Tensor] = None, # K,V来源 B,M,C
             mask: Optional[torch.Tensor] = None,
            # mask:可选注意力掩码，形状与x一致，（但逻辑上应该广播到（B，nh，N,N）），实际使用中通常为 (B, 1, 1, N) 或 (B, N, N)。此处实现假设 mask 已适配
    ) -> torch.Tensor:
        if kv_x is None:
            kv_x = x
        B, N, C = x.shape
        _,M,_ = kv_x.shape
        qkv = []
        for matrix in [self.W_q, self.W_k, self.W_v]:
            temp = matrix(x)  # （B,N,C）
            # 重排多头格式（B,num_heads,N,head_dim）
            temp = rearrange(temp,
                                pattern="b n (nh nd) -> b n nh nd",
                                nh=self.num_heads,
                                nd=self.head_dim
                                )
            qkv.append(temp)
        q, k, v = qkv  # 每个都是（B，nh，N，nd）

         # 可选：对Q和K按head_dim归一化，常用于稳定训练
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 计算缩放点积注意力分数
        # q @ k^T 得到 (B, nh, N, N)，再乘以 1/sqrt(head_dim) 缩放
        attn_scores = q @ k.transpose(-1, -2) * (self.head_dim ** (-0.5))  # (b,nh,n,n)

        if mask is not None:
            # 应用注意力掩码（注意：通常 mask 是加在 softmax 前的负无穷大值）
            #assert mask.shape == x.shape
            attn_scores += mask

        # 对最后一个维度（即 key 维度）做 softmax，得到注意力权重
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        # 加权求和：attn @ V
        y = attn_scores @ v  # (B, nh, N, nd)


        # 将多头结果拼接回 (B, N, C)
        y= rearrange(y,"b nh n nd -> b n (nh nd)")
        y = y.contiguous().view(B, N, self.hidden_dim)  # 保险 reshape
        #y = rearrange(
            #y, pattern="b nh n nd->b n (nh nd)", nh=self.num_heads, nd=self.head_dim
        #)
        y = self.norm(y)
        y = self.out_proj(y)
        y = self.proj_drop(y)

        if self.out_attention:
            return y, attn_scores
        else:
            return y

class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            scale: int, # 隐藏层扩展层数
            proj_drop: float = 0.1,
            qkv_bias: bool = False,
            scale_norm:bool =False,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None, # 设备类型
            dtype=None, # 数据类型
             ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*scale, bias=qkv_bias), # 升维：C → C*scale
            nn.ReLU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim*scale, hidden_dim, bias=qkv_bias), # 降维：C*scale → C
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.net(x) #输入是(B,N,C),输出也是(B,N,C)

class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim:int,
            num_heads: int,
            scale: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype = None,) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        # 默认使用 LayerNorm
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        # 第一个 LayerNorm（用于 Attention 后的残差归一化）
        self.norm1 = norm_layer(hidden_dim, **dd)
        # 第二个 LayerNorm (用于 FFN 后的残差归一化)
        self.norm2 = norm_layer(hidden_dim, **dd)

        # 注意力模块
        self.mha = MultiHeadAttention(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            out_attention=False,
            scale_norm = False,
            norm_layer = norm_layer,
            device = device,
            dtype =dtype,
        )
        # 前馈网络
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            scale=scale,
            proj_drop=proj_drop,  # 复用 proj_drop 作为 FFN 的 dropout
            qkv_bias=qkv_bias,  # 复用 qkv_bias 控制线性层偏置
            device=device,
            dtype=dtype,
        )
        # 共享同一个 dropout（用于两个残差连接之后）
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x:torch.Tensor, # 输入张量，形状为 (B, N, C)
            mask: Optional[torch.Tensor] = None,# 可选注意力掩码，形状需适配为 (B, N, N) 或广播兼容
    ) -> torch.Tensor:

        #注意力分支
        mha_out = self.mha(x, mask=mask)  # (B, N, C)
        mha_out = self.proj_drop(mha_out)
        x = self.norm1(x + mha_out)  # 残差连接 + LayerNorm

        # 前馈网络分支
        ffn_out = self.ffn(x)  # (B, N, C)
        ffn_out = self.proj_drop(ffn_out)
        x = self.norm2(x + ffn_out)  # 残差连接 + LayerNorm

        return x

class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads:int,
            scale:int,
            attn_drop:float = 0.0,
            proj_drop:float=0.0,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            norm_layer:Optional[Type[nn.Module]]=None,
            device=None,
            dtype=None,
        ) -> None:
        super().__init__()
        dd ={"device":device,"dtype":dtype}

        if norm_layer is None:
            norm_layer = nn.LayerNorm

        #三个LayerNorm层:分别用于 self_att,cross_att,ffn后的残差归一化
        self.norm1 = norm_layer(hidden_dim,**dd) # after self-attn
        self.norm2 = norm_layer(hidden_dim,**dd) # after cross-attn
        self.norm3 = norm_layer(hidden_dim,**dd) # after FFN

        #自注意力模块 Q=K=V=tgt
        self.self_att = MultiHeadAttention(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            out_attention = False,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            scale_norm = False,
            norm_layer = norm_layer,
            device = device,
            dtype = dtype,
        )

        #交叉注意力模块 Q=tgt， K=V=enc1
        self.cross_att = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            out_attention=False,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=False,
            norm_layer=norm_layer,
            device=device,
            dtype=dtype,
        )

        #前馈网络
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            scale=scale,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            scale_norm=False,
            norm_layer=norm_layer,
            device=device,
            dtype=dtype,
        )

        #残差连接后的droppout（作用于所有子模块的输出）
        self.proj_drop = nn.Dropout(proj_drop)

        #输出投影层
        self.proj = nn.Linear(hidden_dim,hidden_dim,bias=qkv_bias,**dd)

    def forward(
            self,
            tgt: torch.Tensor, #(B,M,C)
            enc1: torch.Tensor, #(B,N,C)
            tgt_mask:Optional[torch.Tensor] = None,
            enc1_mask:Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        # 输入(B,M,C) -> 输出 (B,M,C)
            self_att_out = self.self_att(tgt, mask=tgt_mask)  # (B,M,C)
            self_att_out = self.proj_drop(self_att_out)
            tgt = self.norm1(tgt+self_att_out)          #残差 （B,M,C）

            #输入 tgt(B,M,C) + enc1(B,N,C) ->(B,M,C)
            cross_att_out = self.cross_att(tgt, kv_x=enc1, mask=enc1_mask)
            tgt = self.norm2(tgt+cross_att_out)

            # (B,M,C)->(B,M,C)
            ffn_out = self.ffn(tgt)  # (B,M,C)
            ffn_out = self.proj_drop(ffn_out)
            tgt = self.norm3(tgt + ffn_out) # 残差(B,M,C)

            out = self.proj(tgt) # (B,M,C)
            return out

class Transformer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            hidden_dim: int,
            num_heads: int,

            scale: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    scale=scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_norm=False,
                    norm_layer=norm_layer,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    scale=scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_norm=False,
                    norm_layer=norm_layer,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def encode(
            self,
            src: torch.Tensor,#输入源序列，形状为（B,N,C）
            src_mask: Optional[torch.Tensor] = None,#编码器注意力掩码，通常为(B,N,N)
             ) -> torch.Tensor:
            x = src
            for block in self.encoder:
                x = block(x, mask=src_mask)
            return x

    def decode(
            self,
            tgt: torch.Tensor,
            enc1: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            enc1_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        for block in self.decoder:
            x = block(x, enc1, tgt_mask=tgt_mask, enc1_mask=enc1_mask)
        return x

    def forward(
            self,
            src: torch.Tensor, #(B,N,C)
            tgt: torch.Tensor, #(B,M,C)
            tgt_mask: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None, #(B,N)
    ) -> torch.Tensor:

        enc1 = self.encode(src, src_mask)  #(B,N,C)
        # (B,1,1,N) -> (B,nh,M,N)
        enc1_mask = None
        if src_key_padding_mask is not None:

            enc1_mask = src_key_padding_mask.unsqueeze(1)
        out = self.decode(tgt, enc1, tgt_mask=tgt_mask, enc1_mask=enc1_mask) #(B,M,C)
        return out

if __name__ == "__main__":
    x = torch.randn((128, 4, 256)) #对于Decoder和Tranformer当做源序列 (B,N,C)
    z = torch.randn((128,8 ,256)) #目标序列tgt,(B,M,C)
    mha = MultiHeadAttention(256, 4, 0.1, out_attention=False)
    ffn = FeedForward(256,4,0.1,qkv_bias=False)
    enc = EncoderLayer(256, 4, 4, norm_layer=nn.LayerNorm)
    dec = DecoderLayer(256,4,4,norm_layer=nn.LayerNorm)
    trans = Transformer(num_layers=4,hidden_dim=256,scale=4,num_heads=4,attn_drop=0.1,proj_drop=0.1)
    y = mha(x)
    y1 = ffn(x)
    y2 = enc(x)
    y3 = dec(z, y2) # (B,M,C)+(B,N,C) -> (B,M,C)
    y4 = trans(x, z) # (B,N,C)+(B,M,C) -> (B,M,C)
    print(y.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(y4.shape)