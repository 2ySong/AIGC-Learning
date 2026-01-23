import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

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
    
model=PatchEmbedding()
output=model(images)
print(output.shape())