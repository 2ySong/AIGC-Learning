import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class encoder(nn.Module):
    def __init__(self, in_channel=3, hidden_dims=[32,64],latent_dim=4):
        super().__init__()

        modules=[]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,out_channels=h_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channel=h_dim

        self.backbone=nn.Sequential(*modules)

        self.projection_layer=nn.Conv2d(in_channels=hidden_dims[-1],out_channels=latent_dim*4,kernel_size=3,padding=1)

    def forward(self,x):
        features=self.backbone(x)
        print(features.shape)
        stat_features=self.projection_layer(features)
        mu,log_var=torch.chunk(stat_features,2,dim=1)
        return mu,log_var

if __name__ == '__main__':
    # 模拟一张 32x32 的 RGB 图片，Batch Size = 2
    dummy_input = torch.randn(2, 3, 32, 32)
    
    encoder = encoder(in_channel=3, hidden_dims=[32, 64], latent_dim=4)
    mu, log_var = encoder(dummy_input)
    
    print(f"Input Shape:   {dummy_input.shape}")
    print(f"Mu Shape:      {mu.shape}")      # 期望: [2, 4, 8, 8]
    print(f"Log_Var Shape: {log_var.shape}") # 期望: [2, 4, 8, 8]
    print("Encoder test passed!")