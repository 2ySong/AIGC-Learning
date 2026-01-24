import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size,num_heads):
        super().__init__()
        self.ln1=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attn=nn.MultiheadAttention(embed_dim=hidden_size,num_heads=num_heads,batch_first=True)
        self.ln2=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp=nn.Sequential(
            nn.Linear(hidden_size,hidden_size*4),
            nn.GELU(),
            nn.Linear(hidden_size*4,hidden_size),
        )
        self.adaLN=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size*6)
        )
        nn.init.constant_(self.adaLN[1].weight,0)
        nn.init.constant_(self.adaLN[1].bias,0)

    def forward(self,x,c):#x是图片，c是时间特征，也就是前文写的TimeEmbedding的输出
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=(self.adaLN(c).chunk(6,dim=1))

        x_norm=modulate(self.ln1(x),shift_msa,scale_msa)
        attn_output=self.attn(x_norm,x_norm,x_norm)[0]
        x=x+gate_msa.unsqueeze(1)*attn_output
        x_norm=modulate(self.ln2(x),shift_mlp,scale_mlp)
        mlp_output=self.mlp(x_norm)
        x=x+gate_mlp.unsqueeze(1)*mlp_output
        return x