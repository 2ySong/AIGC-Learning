import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import math
from einops import rearrange

def rotate_lookup_table(dim,content_length=100,theta=10000):
    rotate_term=1.0/(theta**(torch.arange(0,dim,2)/dim))
    position=torch.arange(0,content_length,1).unsqueeze(1)
    position_lookup_table=position*rotate_term
    position_lookup_table=torch.polar(torch.ones_like(position_lookup_table),position_lookup_table).unsqueeze(0)
    return position_lookup_table

def compute_position(Q,K,position_lookup_table):
    Q=rearrange(Q,'B T H (D complex)-> B T H D complex',complex=2)
    K=rearrange(K,'B T H (D complex)-> B T H D complex',complex=2)
    Q=torch.view_as_complex(Q.float())
    K=torch.view_as_complex(K.float())
    content_length=Q.shape[1]
    position_table=position_lookup_table[:,:content_length,:]
    position_table=rearrange(position_table,'B T (H D)-> B T H D',H=1)
    Q=Q*position_table
    K=K*position_table
    Q=torch.view_as_real(Q).flatten(3)
    K=torch.view_as_real(K).flatten(3)
    return Q.transpose(1,2),K.transpose(1,2)