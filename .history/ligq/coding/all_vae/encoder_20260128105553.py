import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms



class VAE(nn.Module):
    def __init__(self,hidden_dim=[32,64],in_channels=3,latent_channel=4):
        super().__init__()
        modules=[]
        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                   nn.Conv2d(in_channels,h_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels=h_dim
        self.encoder=nn.Sequential(*modules)

        self.mu_logvar=nn.Conv2d(hidden_dim[-1],latent_channel*2,kernel_size=3,padding=1)

        self.decoder_input=nn.Conv2d(latent_channel,hidden_dim[-1],kernel_size=3,padding=1)
        hidden_dim.reverse()
        modules=[]
        for i in range(len(hidden_dim)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i],hidden_dim[i+1],kernel_size=3,stride=2,padding=1,output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder=nn.Sequential(*modules)

        self.final_layer=nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1],hidden_dim[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1],out_channels=3,kernel_size=3,padding=1),
            nn.Sigmoid()
        )

    def encode(self,x):
        result=self.encoder(x)
        mu_logvar=self.mu_logvar(result)
        mu,log_var=torch.chunk(mu_logvar,2,dim=1)
        return mu,log_var
    
    def reparameterize(self,mu,log_var):
        if self.training:
            std=torch.exp(0.5*log_var)
            eps=torch.randn_like(std)
            return mu+eps*std
        else:
            return mu
        
    def decode(self,z):
        result=self.decoder_input(z)
        result=self.decoder(result)
        result=self.final_layer(result)
        return result
        
    def forward(self,x):
        mu,log_var=self.encode(x)
        z=self.reparameterize(mu,log_var)
        recon=self.decode(z)
        return recon,mu,log_var




