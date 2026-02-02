import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class VAE(nn.Module):
    def __init__(self, in_channel=3,hidden_dims=[32,64],latent_dim=4):
        super().__init__()
        self.hidden_dims=hidden_dims
        self.latent_dim=latent_dim
        self.in_channel=in_channel
        modules=[]
        current_in_channel=in_channel
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=current_in_channel,out_channels=h_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            current_in_channel=h_dim
        self.encoder=nn.Sequential(*modules)

        self.mu_log_var=nn.Conv2d(hidden_dims[-1],latent_dim*2,kernel_size=3,padding=1)

        decoder_hidden_dim=hidden_dims[::-1]

        self.decoder_input=nn.Conv2d(latent_dim,hidden_dims[-1],kernel_size=3,padding=1)
        modules=[]
        for i in range(len(decoder_hidden_dim)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_hidden_dim[i],decoder_hidden_dim[i+1],kernel_size=3,stride=2,padding=1,output_padding=1),
                    nn.BatchNorm2d(decoder_hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder=nn.Sequential(*modules)

        self.final_layer=nn.Sequential(
            nn.ConvTranspose2d(decoder_hidden_dim[-1],decoder_hidden_dim[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(decoder_hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_hidden_dim[-1],in_channel,kernel_size=3,padding=1),
            nn.Sigmoid()
        )
    
    def encode(self,x):
        result=self.encoder(x)
        result=self.mu_log_var(result)
        mu,log_var=torch.chunk(result,2,dim=1)
        return mu,log_var
    
    def reparameter(self,mu,log_var):
        if self.training:
            eps=torch.randn_like(mu)
            std=torch.exp(0.5*log_var)
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
        x=self.reparameter(mu,log_var)
        recon_x=self.decode(x)
        return recon_x,mu,log_var
    
def loss(recon_x,x,mu,log_var,kld_weight):
    recon_loss=F.mse_loss(recon_x,x,reduction='sum')
    kld_loss=torch.mean(-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp(),dim=1),dim=0)
    loss=recon_loss+kld_loss*kld_weight
    return loss
    

