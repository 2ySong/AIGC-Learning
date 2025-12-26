import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,dim,head_nums,dropout=0.1):
        super().__init__()
        self.dim=dim
        self.head_nums=head_nums
        assert dim%head_nums==0
        self.head_dim=dim//head_nums
        self.q=nn.Linear(dim,dim)
        self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)
        self.proj=nn.Linear(dim,dim)
    def forward(self,x,y,z,mask=None):
        B=x.shape[0]
        q=self.q(x).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)
        k=self.k(y).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)
        v=self.v(z).reshape(B,-1,self.head_nums,self.head_dim).transpose(1,2)

        att=q@k.transpose(-2,-1)/self.head_dim**0.5
        if mask is not None:
            if mask.dim==3:
                mask=mask.unsqueeze(0)
            att=att.masked_fill(mask==0,-float('inf'))
        att=F.softmax(att,dim=-1)
        att=self.dropout(att)
        att=att@v
        att=att.transpose(1,2).reshape(B,-1,self.dim)
        return self.proj(att)
class FeedForward(nn.Module):
    def __init__(self,dim,scale,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,dim*scale),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*scale,dim)
        )
    def forward(self,x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.att=MultiHeadAttention(dim,head_nums,dropout=dropout)
        self.ffn=FeedForward(dim,scale,dropout)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        x_=self.dropout(self.att(x,x,x,mask=mask))
        x=self.norm1(x_+x)
        x_=self.dropout(self.ffn(x))
        x=self.norm2(x_+x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.self_att=MultiHeadAttention(dim,head_nums,dropout=dropout)
        self.cross_att=MultiHeadAttention(dim,head_nums,dropout=dropout)
        self.ffn=FeedForward(dim,scale,dropout)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.norm3=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,tgt,enc,tgt_mask=None,enc_mask=None):
        x_=self.dropout(self.self_att(tgt,tgt,tgt,mask=tgt_mask))
        tgt=self.norm1(x_+tgt)
        x_=self.dropout(self.cross_att(tgt,enc,enc,mask=enc_mask))
        tgt=self.norm2(x_+tgt)
        x_=self.dropout(self.ffn(tgt))
        out=self.norm2(x_+tgt)
        return out

class Transformer(nn.Module):
    def __init__(self,layer_nums,dim,head_nums,scale,dropout=0.1):
        super().__init__()
        self.encoder=nn.ModuleList(
            [EncoderLayer(dim,head_nums,scale,dropout) for _ in range(layer_nums)]
        )
        self.decoder=nn.ModuleList(
            [DecoderLayer(dim,head_nums,scale,dropout) for _ in range(layer_nums)]
        )
        self.proj = nn.Linear(dim, dim)
    def encode(self,x,mask=None):
        for block in self.encoder:
            x=block(x,mask)
        return x
    def decode(self,tgt,enc,tgt_mask=None,enc_mask=None):
        for block in self.decoder:
            tgt=block(tgt,enc,tgt_mask=tgt_mask,enc_mask=enc_mask)
        return tgt
    def forward(self,src,tgt,tgt_mask=None,enc_mask=None):
        enc=self.encode(src,mask=enc_mask)
        out=self.decode(tgt,enc,tgt_mask=tgt_mask,enc_mask=enc_mask)
        return self.proj(out)
if __name__=="__main__":
    B,H,W=2,4,256
    head_nums=4
    x=torch.rand((B,H,W))
    mask=torch.ones((B,head_nums,H,H))

    model=MultiHeadAttention(W,head_nums=head_nums)
    print(model(x,x,x,mask=mask).shape)
    model=FeedForward(W,4)
    print(model(x).shape)
    model=EncoderLayer(dim=W,head_nums=head_nums,scale=4,)
    print(model(x).shape)
    model=DecoderLayer(dim=W,head_nums=head_nums,scale=4)
    print(model(x,x).shape)
    model=Transformer(layer_nums=4,dim=W,head_nums=head_nums,scale=4)
    print(model(x,x).shape)

