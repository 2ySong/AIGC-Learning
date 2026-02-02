import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import  torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel=3,hidden_dim=128,patch_size=4,img_size=32):
        super().__init__()
        self.convolution=nn.Conv2d(in_channels=in_channel,out_channels=hidden_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x=self.convolution(x)
        x.flatten(2)
        x.transpose(1,2)
        return x
    
class TimeEmbedding(nn.Module):
    def __init__(self, freq_dim=256,hidden_dim=128):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(freq_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.freq_dim=freq_dim
        self.hidden_dim=hidden_dim
    
    @staticmethod
    def TimeTable(self,t,freq=10000):
        t=t.unsqueeze(1)
        div_term=torch.exp(-math.log(freq)*torch.arange(0,self.freq_dim,2)/self.freq_dim)
        timetable=t*div_term
        timetable=torch.cat([torch.cos(timetable),torch.sin(timetable)],dim=1)
        return timetable
    
    def forward(self,t):
        timetable=self.TimeTable(t)
        timetable=self.mlp(timetable)
        return timetable
    
def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=128,num_head=4):
        super().__init__()
        self.ln1=nn.LayerNorm(hidden_dim,eps=1e-6,elementwise_affine=False)
        self.ln2=nn.LayerNorm(hidden_dim,eps=1e-6,elementwise_affine=False)
        self.multihead=nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_head)
        self.adaLN=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim*6)
        )
        self.mlp=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4,hidden_dim)
        )
        nn.init.constant_(self.adaLN[1].weight,0)
        nn.init.constant_(self.adaLN[1].bias,0)
    def forward(self,x,c):
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=(self.adaLN(c).chunk(6,dim=1))
        x_norm=modulate(self.ln1(x),shift=shift_msa,scale=scale_msa)
        attn=self.multihead(x_norm,x_norm,x_norm)[0]
        x=x+attn*gate_msa.unsqueeze(1)
        x_norm=modulate(self.ln2(x),shift=shift_mlp,scale=scale_mlp)
        mlpresult=self.mlp(x_norm)
        x=x+mlpresult*gate_mlp.unsqueeze(1)

        return x
    
class Finallayer(nn.Module):
    def __init__(self, hidden_dim=128,patch_size=32,in_channels=3):
        super().__init__()
        self.ln=nn.LayerNorm(hidden_dim,eps=1e-6,elementwise_affine=False)
        self.adaLN=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim*2)
        )
        nn.init.constant_(self.adaLN[1].weight,0)
        nn.init.constant_(self.adaLN[1].bias,0)
        self.finallinear=nn.Linear(hidden_dim,in_channels*patch_size*patch_size)

    def forward(self,x,c):
        shift,scale=(self.adaLN(c).chunk(2,dim=1))
        x=modulate(self.ln(x),shift,scale)
        x=self.finallinear(x)
        return x
    
class DiT(nn.Module):
    def __init__(self, in_channel=3,hidden_dim=128,patch_size=4,img_size=32,num_head=4,floor=4):
        super().__init__()
        self.in_channel=in_channel
        self.patch_num=(img_size//patch_size)**2
        self.PatchEmbed=PatchEmbedding(in_channel,hidden_dim,patch_size,img_size)
        self.TimeEmbed=TimeEmbedding(hidden_dim=hidden_dim)
        self.PosiEmbed=nn.Parameter(torch.zeros(1,self.patch_num,hidden_dim))

        self.Blocks=nn.ModuleList([DiTBlock(hidden_dim,num_head)for _ in range (floor)])
        self.FinalLayer=Finallayer(hidden_dim,patch_size,in_channel)

        nn.init.normal_(self.PosiEmbed,0,0.02)
    
    def unpatchify(self,x):
        x=rearrange(x,'B (P P) (c p p) -> B c (P p) (P p)',c=self.in_channel)
        return x

    def forward(self,x,t):
        c=self.TimeEmbed(t)
        x=self.PatchEmbed(x)
        x=x+self.PosiEmbed
        for block in self.Blocks:
            x=block(x,c)
        x=self.FinalLayer(x,c)
        x=self.unpatchify(x)
        return x

def get_loss(model,images):
    batch_size=images[0]
    t=torch.rand(batch_size)
    noise=torch.rand_like(images)
    t_reshape=rearrange(t,'(B l l l)->B l l l',l=1)
    x_t=(1-t_reshape)*noise+t_reshape*images
    predict=model(x_t,t)
    target=images-noise
    loss=F.mse_loss(predict,target)
    return loss
