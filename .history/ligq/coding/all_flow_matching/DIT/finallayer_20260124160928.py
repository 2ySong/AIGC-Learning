import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import math
import einops
from einops import rearrange

def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size,patch_size,out_channels):
        super().__init__()
        self.ln_final=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.linear=nn.Linear(hidden_size,patch_size*patch_size*out_channels)
        self.adaLN=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size*2)
        )
        nn.init.constant_(self.adaLN[1].weight,0)
        nn.init.constant_(self.adaLN[1].bias,0)
    
    def forward(self,x,c):
        shift,scale=(self.adaLN(c).chunk(2,dim=1))
        x=modulate(self.ln_final(x),shift,scale)
        x=self.linear
        return x
    
class DiT(nn.Module):
    def __init__(self, image_size=32,patch_size=4,in_channels=3,hidden_size=128,depth=6,num_heads=4):
        super().__init__()

        self.image_size=image_size
        self.patch_size=patch_size
        self.in_channels=in_channels

        num_patch=(image_size//patch_size)**2

        self.patch_embed=PatchEmbedding(in_channels,hidden_size,image_size,patch_size)
        self.time_embed=TimeTable(256,hidden_size)
        self.posi_embed=nn.Parameter(torch.zeros(1,num_patch,hidden_size))

        self.blocks=nn.ModuleList([DiTBlock(hidden_size,num_heads) for _ in range(depth)])
        self.final_layer=FinalLayer(hidden_size,patch_size,in_channels)
        nn.init.normal_(self.posi_embed,std=0.02)

    def unpatchify(self,x):
        imgs = rearrange(x,'B (P P) (p p c) -> B c (P p) (P p)',p=self.patch_size,P=self.image_size//self.patch_size)
        return imgs
    
    def forward(self,x,t):
        x=self.patch_embed(x)
        x=x+self.posi_embed
        c=self.time_embed(t)
        for block in self.blocks:
            x=block(x,c)
        x=self.final_layer(x,c)
        x=self.unpatchify(x)
        return x