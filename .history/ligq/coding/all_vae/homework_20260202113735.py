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

        self.projection_layer=nn.Conv2d(in_channels=hidden_dims[-1],out_channels=latent_dim*2,kernel_size=3,padding=1)

    def forward(self,x):
        features=self.backbone(x)
        stat_features=self.projection_layer(features)
        print(stat_features.shape)
        mu,log_var=torch.chunk(stat_features,2,dim=1)
        return mu,log_var

"""if __name__ == '__main__':
    # 模拟一张 32x32 的 RGB 图片，Batch Size = 2
    dummy_input = torch.randn(2, 3, 32, 32)
    
    encoder = encoder(in_channel=3, hidden_dims=[32, 64], latent_dim=4)
    mu, log_var = encoder(dummy_input)
    
    print(f"Input Shape:   {dummy_input.shape}")
    print(f"Mu Shape:      {mu.shape}")      # 期望: [2, 4, 8, 8]
    print(f"Log_Var Shape: {log_var.shape}") # 期望: [2, 4, 8, 8]
    print("Encoder test passed!")"""

def reparameterize(self,mu,log_var):
    if self.training:
        std=torch.exp(0,5*log_var)
        eps=torch.randn_like(std)
        return mu+eps*std
    else:
        return mu  

class Decoder(nn.Module):
    def __init__(self, latent_dim=4, hidden_dims=[32,64]):
        super().__init__()      
        self.hidden_dims=hidden_dims[::-1]
        self.decoder_input=nn.Conv2d(in_channels=latent_dim,out_channels=hidden_dims[0],kernel_size=3,padding=1)

        modules=[]
        for i in range(len(self.hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.hidden_dims[i],out_channels=self.hidden_dims[i+1],kernel_size=3,padding=1,stride=2,output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.backbone=nn.Sequential(*modules)
        
        self.final_layer=nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],self.hidden_dims[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1],out_channels=3,kernel_size=3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        result=self.decoder_input(x)
        result=self.backbone(result)
        result=self.final_layer(result)

        return result
    
def vae_loss_function(recon_x,x,mu,log_var,kld_weight=0.00025):
    recon_loss=F.mse_loss(recon_x,x,reduction='sum')
    kld_loss=-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
    loss=recon_loss+kld_loss