import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, dim,num_heads,dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.head_dim=dim//num_heads
        self.dropout = nn.Dropout(dropout)
        self.q=nn.Linear(dim,dim)
        self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)
        self.proj=nn.Linear(dim,dim)
    def forward(self,x,y,z,mask=None):
        q=self.q(x).reshape(x.shape[0],-1,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k(y).reshape(y.shape[0],-1,self.num_heads,self.head_dim).transpose(1,2)
        v=self.v(z).reshape(z.shape[0],-1,self.num_heads,self.head_dim).transpose(1,2)
        att=q@k.transpose(-2,-1)/self.head_dim**0.5
        if mask is not None:
            att=att.masked_fill(mask==0,-float('inf'))
        att=F.softmax(att,dim=-1)
        att=self.dropout(att)
        att=(att@v).transpose(1,2).reshape(x.shape[0],-1,self.dim)
        return self.proj(att)
class FeedForward(nn.Module):
    def __init__(self, dim, scale=4, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim, dim*scale),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*scale, dim),
        )
    def forward(self,x):
        return self.net(x)
class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, scale, dropout=0.1):
        super().__init__()
        self.att=MultiHeadAttention(dim, num_heads,dropout)
        self.ffn=FeedForward(dim,scale,dropout)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        x=self.norm1(x+self.dropout(self.att(x,x,x,mask)))
        x=self.norm2(self.dropout(self.ffn(x))+x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,dim, num_heads, scale, dropout=0.1):
        super().__init__()
        self.satt=MultiHeadAttention(dim,num_heads,dropout)
        self.cross_att=MultiHeadAttention(dim,num_heads,dropout)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.norm3=nn.LayerNorm(dim)
        self.ffn=FeedForward(dim,scale,dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,tgt,enc_out,tgt_mask=None,enc_mask=None):
        tgt=self.norm1(tgt+self.dropout(self.satt(tgt,tgt,tgt,tgt_mask)))
        tgt=self.norm2(tgt+self.dropout(self.cross_att(tgt,enc_out,enc_out,enc_mask)))
        out=self.norm3(tgt+self.dropout(self.ffn(tgt)))
        return out
class PositionEncoding(nn.Module):
    def __init__(self,dim,max_len=5000):
        super().__init__()
        pe=torch.zeros((max_len,dim))
        position=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000)/dim))
        pe[:,::2]=torch.sin(position*div)
        pe[:,1::2]=torch.cos(position*div)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return x+self.pe[:,:x.size(1),:].to(x.device)
class Transformer(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,layer_nums,dim,max_len,num_heads, scale, dropout):
        super().__init__()
        self.encoder=nn.ModuleList([
            EncoderLayer(dim,num_heads, scale, dropout) for _ in range(layer_nums)
        ])
        self.decoder=nn.ModuleList([
            DecoderLayer(dim,num_heads, scale, dropout) for _ in range(layer_nums)
        ])
        self.proj=nn.Linear(dim,tgt_vocab)
        self.tgt_embed=nn.Embedding(tgt_vocab,dim)
        self.src_embed=nn.Embedding(src_vocab,dim)
        self.pos=PositionEncoding(dim,max_len)
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        src=self.pos(self.src_embed(src))
        tgt=self.pos(self.tgt_embed(tgt))
        for layer in self.encoder:
            src=layer(src,src_mask)
        for layer in self.decoder:
            tgt=layer(tgt,src,tgt_mask,enc_mask=src_mask)
        return self.proj(tgt)
if __name__=='__main__':
    src_vocab, tgt_vocab, layer_nums=10000,10000,2
    dim, max_len, num_heads, scale=256,5000,4,2
    model=Transformer(
        src_vocab, tgt_vocab, layer_nums,
        dim, max_len, num_heads, scale, dropout=0.1
    )
    batch_size=2
    length=100
    src=torch.randint(0,src_vocab,(batch_size,length))
    tgt=torch.randint(0,tgt_vocab,(batch_size,length))
    src_mask=(src!=0).unsqueeze(1).unsqueeze(2)
    pad_mask=(tgt!=0).unsqueeze(1).unsqueeze(2)
    tgt_mask=torch.tril(torch.ones(length,length,dtype=torch.bool))
    tgt_mask=tgt_mask.unsqueeze(0).unsqueeze(1)
    tgt_mask=tgt_mask&pad_mask
    print(model(tgt,src,src_mask,tgt_mask).shape)
