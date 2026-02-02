import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import  torch.nn.functional as F

def get_loss(model,images):
    batch_size=images.shape[0]
    t=torch.rand(batch_size)
    noise=torch.randn_like(images)
    t_reshape=rearrange(t,'(batch_size d d d)->batch_size d d d',d=1)
    x_t=(1-t_reshape)*noise+t_reshape*images
    target=images-noise
    predict=model(x_t,t)
    loss=F.mse_loss(predict,target)
    return loss

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
dataset=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)
dataloader=torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=True)
images,labels=next(iter(dataloader))

class PatchEmbedding(nn.Module):
    def __init__(self, in_size=3, out_size=128,img_size=32,patch_size=8):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.convolution=nn.Conv2d(in_channels=in_size,out_channels=out_size,stride=patch_size,kernel_size=patch_size)

    def forward(self,x):
        x=self.convolution(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
    
#model=PatchEmbedding()
#output=model(images)
#print(output.shape)
class TimeEmbedding(nn.Module):
    def __init__(self, freq_number=256, hidden_dim=128):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(freq_number,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.freq_number=freq_number
        self.hidden_dim=hidden_dim
    
    def TimeTable(self,t,freq=10000):
        time=t.unsqueeze(1)
        div_term=torch.exp(-math.log(freq)*torch.arange(0,self.freq_number,2)/self.freq_number)
        timetable=time*div_term
        timetable=torch.cat([torch.cos(timetable),torch.sin(timetable)],dim=-1)
        return timetable
    
    def forward(self,t):
        embedding=self.TimeTable(t)
        embedding=self.mlp(embedding)

        return embedding
    
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

    def forward(self,x,c):
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=(self.adaLN(c).chunk(6,dim=1))
        x_norm=modulate(self.ln1(x),shift_msa,scale_msa)
        attn=self.attn(x_norm,x_norm,x_norm)[0]
        x=x+attn*gate_msa.unsqueeze(1)

        x_norm=modulate(self.ln2(x),shift_mlp,scale_mlp)
        mlp_term=self.mlp(x_norm)
        x=x+mlp_term*gate_mlp.unsqueeze(1)

        return x
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size,patch_size,in_channels):
        super().__init__()
        self.final_ln=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.linear=nn.Linear(hidden_size,patch_size*patch_size*in_channels)
        self.adaLN=nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size*2)
        )
        nn.init.constant_(self.adaLN[1].weight,0)
        nn.init.constant_(self.adaLN[1].bias,0)
    def forward(self,x,c):
        shift,scale=(self.adaLN(c).chunk(2,dim=1))
        x=modulate(self.final_ln(x),shift,scale)
        return self.linear(x)
    
class DiT(nn.Module):
    def __init__(self, img_size=32,patch_size=4,hidden_size=128,in_channels=3,num_heads=4,floor=6):
        super().__init__()
        self.patch_size=patch_size
        self.img_size=img_size
        self.num_patch=(img_size//patch_size)**2
        self.time_embed=TimeEmbedding(freq_number=256,hidden_dim=hidden_size)
        self.patch_embed=PatchEmbedding(in_size=in_channels,out_size=hidden_size,img_size=img_size,patch_size=patch_size)
        self.posi_embed=nn.Parameter(torch.zeros(1,self.num_patch,hidden_size))

        self.blocks=nn.ModuleList([DiTBlock(hidden_size,num_heads)for _ in range(floor)])
        self.final_layer=FinalLayer(hidden_size,patch_size,in_channels)
        nn.init.normal_(self.posi_embed,std=0.02)

    def unpatchify(self,x):
        img=rearrange(x,'B (P P) (p p c)-> B c (P p) (P p)',P=self.img_size//self.patch_size,p=self.patch_size)
        return img
    
    def forward(self,x,t):
        c=self.time_embed(t)
        x=self.patch_embed(x)
        x=x+self.posi_embed
        for block in self.blocks:
            x=block(x,c)
        x=self.final_layer(x,c)
        img=self.unpatchify(x)
        return img

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])
dataset=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=2)
model=DiT(img_size=32,patch_size=4,hidden_size=128,floor=6).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
epochs=5
print('start training')

for epoch in range(epochs):
    model.train()
    total_loss=0
    for i,(imgs,labels) in enumerate(dataloader):
        imgs=imgs.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss=get_loss(model,imgs)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    
    avg_loss=total_loss/len(dataloader)
    print(f'epoch{epoch}finished,average loss={avg_loss:.4f}')
print("finished training")