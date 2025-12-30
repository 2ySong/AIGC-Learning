import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):#多头注意力机制
    def __init__(self,dim,head_nums,dropout):
        super().__init__()
        self.dim=dim
        self.head_nums=head_nums
        assert dim%head_nums==0
        self.head_dim=dim//head_nums
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x,y,z,mask=None):
        B=x.shape[0]
        q=self.q(x).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)
        k=self.k(y).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)
        v=self.v(z).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)

        att=q@k.transpose(-2,-1)/self.head_dim**0.5
        if mask is not None:# 应用掩码（用于处理填充或未来信息）
            att = att.masked_fill(mask==0,-float('inf'))
        att=F.softmax(att,dim=-1)
        att=self.dropout(att)
        att=att@v
        att=att.transpose(1,2).reshape(B,-1,self.dim)
        return self.proj(att)

class FeedForward(nn.Module):#前馈神经网络
    def __init__(self,dim,scale,dropout=0.1):
        super().__init__()
        # 两层全连接，中间放大 scale 倍
        self.net=nn.Sequential(
            nn.Linear(dim,dim*scale),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*scale,dim)
        )
    def forward(self,x):
        return self.net(x)

class EncoderLayer(nn.Module):# Transformer 编码层
    def __init__(self,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)
        self.att = MultiHeadAttention(dim=dim, head_nums=head_nums, dropout=dropout)
        self.ffn = FeedForward(dim=dim, scale=scale, dropout=dropout)
    def forward(self,x,mask=None):
        x_=self.dropout(self.att(x,x,x,mask=mask))# 子层 1：多头自注意力 + Add&Norm
        x=self.norm1(x_+x)
        x_=self.dropout(self.ffn(x))# 子层 2：前馈网络 + Add&Norm
        x=self.norm2(x_+x)
        return x

class DecoderLayer(nn.Module):# Transformer 解码层
    def __init__(self,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.self_att = MultiHeadAttention(dim,head_nums,dropout=dropout)# 1. Masked 自注意力
        self.cross_att = MultiHeadAttention(dim,head_nums,dropout=dropout)# 2. 编码-解码交叉注意力
        self.ffn=FeedForward(dim,scale,dropout)# 3. 前馈网络
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.norm3=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)
        self.proj=nn.Linear(dim,dim)# 最后可选投影

    def forward(self,tgt,enc,tgt_mask=None,enc_mask=None):
        # 子层 1：Masked 自注意力
        x_=self.dropout(self.self_att(tgt,tgt,tgt,mask=tgt_mask))
        tgt=self.norm1(x_+tgt)
        # 子层 2：交叉注意力（Query 来自解码端，K/V 来自编码端）
        x_=self.dropout(self.cross_att(tgt,enc,enc,mask=enc_mask))
        tgt=self.norm2(x_+tgt)
        # 子层 3：前馈
        x_= self.dropout(self.ffn(tgt))
        out =self.norm3(x_+tgt)
        return self.proj(out)

class Transformer(nn.Module):
    def __init__(self,layer_nums,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.encoder=nn.ModuleList([EncoderLayer(dim,head_nums,scale,dropout) for _ in range(layer_nums)])
        self.decoder=nn.ModuleList([DecoderLayer(dim,head_nums,scale,dropout) for _ in range(layer_nums)])
    def encode(self,x,mask=None):
        for block in self.encoder:
            x=block(x,mask)
        return x
    def decode(self,tgt,enc,tgt_mask=None,enc_mask=None):
        for block in self.decoder:
            tgt=block(tgt,enc,tgt_mask,enc_mask=None)
        return tgt
    def forward(self,src,tgt,tgt_mask=None,enc_mask=None):
        enc=self.encode(src,mask=enc_mask)
        out=self.decode(tgt,enc,tgt_mask=None,enc_mask=None)
        return out




if __name__=="__main__":
    B,H,W=2,4,256
    x=torch.randn((B,H,W))
    mask=torch.rand((B,W,W))
    head_nums=4
    model=MultiHeadAttention(W,head_nums=head_nums,dropout=0.1)
    print(model(x,x,x).shape)
    model=FeedForward(W,4)
    print(model(x).shape)
    model=EncoderLayer(dim=W,head_nums=head_nums,scale=4)
    print(model(x).shape)
    model=DecoderLayer(dim=W,head_nums=head_nums,scale=4)
    print(model(x,x).shape)
    model=Transformer(layer_nums=4,dim=W,head_nums=head_nums,scale=4)
    print(model(x,x).shape)