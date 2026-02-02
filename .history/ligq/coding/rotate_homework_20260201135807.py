import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import math
from einops import rearrange

def rotate_table(dim,content_length,freq=10000):
    position=torch.arange(0,content_length,1).unsqueeze(1)
    div_term=torch.exp(-math.log(freq)*torch.arange(0,dim,2)/dim)
    position_table=position*div_term
    position_table=torch.polar(torch.ones_like(position_table),position_table).unsqueeze(0)
    return position_table

def compute_position(Q,K,position_table):
    Q=rearrange(Q,'B T H (D complex)->B T H D complex',complex=2)
    K=rearrange(K,'B T H (D complex)->B T H D complex',complex=2)
    content_length=Q.shape[1]
    position_table=position_table[:,:content_length,:]
    position_table=position_table.unsqueeze(2)
    Q=torch.view_as_complex(Q.float())
    K=torch.view_as_complex(K.float())
    Q=Q*position_table
    K=K*position_table
    Q=torch.view_as_real(Q)
    K=torch.view_as_real(K)
    Q=rearrange(Q,'B T H D complex->B T H (D complex)',complex=2)
    K=rearrange(K,'B T H D complex->B T H (D complex)',complex=2)
    return Q.transpose(1,2),K.transpose(1,2)