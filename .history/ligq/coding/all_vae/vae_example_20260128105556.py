import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, hidden_dims=[32, 64]):
        super(VAE, self).__init__()
        self.latent_channels = latent_channels

        # ================= Encoder (压缩) =================
        # 目标：将 3x32x32 -> 压缩为 latent_channels x 8 x 8
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # 核心层：预测均值(mu)和方差的对数(log_var)
        # 这里的输出通道是 2 * latent_channels，因为要切分成 mu 和 log_var
        self.fc_mu_logvar = nn.Conv2d(hidden_dims[-1], latent_channels * 2, kernel_size=3, padding=1)

        # ================= Decoder (解压) =================
        # 目标：将 latent_channels x 8 x 8 -> 还原为 3x32x32
        modules = []
        # 先把 latent 映射回 hidden_dims[-1]
        self.decoder_input = nn.Conv2d(latent_channels, hidden_dims[-1], kernel_size=3, padding=1)
        
        hidden_dims.reverse() # [64, 32]
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], 
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        
        # 最后一层：还原回 RGB 3通道
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid() # 输出范围限制在 0-1 (匹配 ToTensor)
        )

    def encode(self, x):
        """
        编码过程：
        x: [B, 3, 32, 32] -> result: [B, 64, 8, 8]
        """
        result = self.encoder(x)
        
        # 映射到 mu 和 log_var
        # mu_logvar: [B, 2*latent, 8, 8]
        mu_logvar = self.fc_mu_logvar(result)
        
        # 在通道维度切分: mu=[B, 4, 8, 8], log_var=[B, 4, 8, 8]
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧 (The Reparameterization Trick)
        z = mu + std * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * log_var) # 也就是 sigma
            eps = torch.randn_like(std)    # 从标准正态分布 N(0, I) 采样噪音
            return mu + eps * std
        else:
            # 推理阶段直接用均值，保证确定性 (也可以采样，看需求)
            return mu

    def decode(self, z):
        """
        解码过程：
        z: [B, 4, 8, 8] -> recon: [B, 3, 32, 32]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

# ================= Loss 函数 (核心) =================
def loss_function(recon_x, x, mu, log_var, kld_weight=0.00025):
    """
    Loss = 重建误差 (MSE) + KL散度 (KLD)
    """
    # 1. 重建误差：生成的图和原图像不像？
    # 也可以用 F.binary_cross_entropy，如果图片是 0-1
    recons_loss = F.mse_loss(recon_x, x) 
    
    # 2. KL 散度：分布是否接近标准正态分布 N(0, I)？
    # 公式推导结果: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 这里的 sum 是对 Batch 和 Dimensions 求和，最后通常取 mean
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
    
    # 3. 总 Loss
    loss = recons_loss + kld_weight * kld_loss.mean()
    return loss, recons_loss, kld_loss.mean()

# ================= 简单的训练测试 =================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 64
    lr = 0.001
    epochs = 10
    latent_dim = 4 # 潜变量通道数

    # 数据集
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), # 0-1 范围
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    vae = VAE(in_channels=3, latent_channels=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # 训练循环
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # Forward
            recon_imgs, mu, log_var = vae(imgs)
            
            # Compute Loss
            loss, recon_loss, kld_loss = loss_function(recon_imgs, imgs, mu, log_var)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Recon: {total_recon/len(dataloader):.4f} | KLD: {total_kld/len(dataloader):.4f}")

    print("VAE Training Finished!")
    
    # 验证维度 (给你的 DiT 用)
    sample_img = torch.randn(1, 3, 32, 32).to(device)
    mu, log_var = vae.encode(sample_img)
    z = vae.reparameterize(mu, log_var)
    print("\n--- Latent Shape Check ---")
    print(f"Original Image: {sample_img.shape}")
    print(f"Latent Representation (z): {z.shape}") 
    # 预期输出: [1, 4, 8, 8]。这就是你要喂给 DiT 的东西。