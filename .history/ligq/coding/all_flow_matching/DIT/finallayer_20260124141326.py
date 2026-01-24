import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import math

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
    