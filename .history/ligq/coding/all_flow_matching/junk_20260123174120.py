import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import math

device='cuda'

class TimeStepEmbed(nn.Module):
    def __init__(self, freq_number=256, hidden_size=128 ):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(freq_number,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size),
        )
        self.freq_number=freq_number

    @staticmethod
    def timestep_embedding(t,dim,max_period=10000):
        half=dim//2
        freqs=torch.exp(-math.log(max_period)*torch.arange(0,half).to(device=device)/half)
        args=t.unsqueeze(1)*freqs
        embedding=torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
        return embedding

    def forward(self,t):
        embedding=self.timestep_embedding(t,self.freq_number)
        embedding=self.mlp(embedding)

        return embedding