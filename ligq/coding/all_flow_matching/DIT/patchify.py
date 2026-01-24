import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================
# 1. 准备数据 (CIFAR-10)
# ==========================================
# 定义一个转换，把 PIL 图片变成 PyTorch 张量
transform = transforms.Compose([
    transforms.Resize((32, 32)), # 确保是 32x32
    transforms.ToTensor(),       # 变成张量 [C, H, W]，数值 0-1 之间
])

# 下载并加载 CIFAR-10 测试集 (只需要几张图来测试)
# root='./data' 指定下载路径，如果已存在则不会重复下载
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# 创建一个加载器，batch_size=1，因为我们只想看一张图
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=True)

# 获取一张图片
dataiter = iter(testloader)
images, labels = next(dataiter)
# images 的形状现在是 [1, 3, 32, 32]

# --- 可视化函数 ---
def imshow(img_tensor, title=None):
    # 将张量转回 numpy 数组用于显示
    img = img_tensor.numpy()
    # PyTorch 是 [C, H, W], Matplotlib 需要 [H, W, C]
    plt.imshow(np.transpose(img, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off') # 不显示坐标轴

# 显示原始图片
plt.figure(figsize=(4, 4))
imshow(images[0], title=f"Input CIFAR Image (32x32)")
plt.show()


# ==========================================
# 2. 定义 Patch Embedding 模型 (和之前一样，稍作修改适配CIFAR)
# ==========================================
class PatchEmbed(nn.Module):
    # 修改默认参数：in_chans=3 (CIFAR是RGB), patch_size=8 (为了演示)
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算总共有多少个方块: (32 // 8) * (32 // 8) = 4 * 4 = 16 个
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size, # 卷积核大小 = 8x8
            stride=patch_size       # 步长 = 8，一步跨过一个patch
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.proj(x)
        # x: [B, 128, 4, 4] -> (32/8=4)
        
        x = x.flatten(2)
        # x: [B, 128, 16]
        
        x = x.transpose(1, 2)
        # x: [B, 16, 128] -> [Batch, 序列长度, 向量维度]
        return x

# ==========================================
# 3. 运行模型
# ==========================================
# 实例化模型
model = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=128)

# 将真实的 CIFAR 图片送入模型
output_sequence = model(images)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # MLP: 用于对正弦波特征进行进一步的学习和变换
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(), # 激活函数，DiT/GPT 常用 SiLU 而不是 ReLU
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        这里是纯数学计算，没有可学习参数。
        把 t (形状 [Batch]) 变成 正弦波向量 (形状 [Batch, dim])
        """
        # 1. 计算频率一半的维度 (因为要有 sin 和 cos 两个)
        half = dim // 2
        
        # 2. 生成频率系数 (freqs)
        # 这一行看着吓人，其实就是生成一串从 1 到 1/10000 的数字
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        # 3. 把 t 和 频率相乘
        # args 形状: [Batch, half_dim]
        args = t[:, None].float() * freqs[None]
        
        # 4. 拼接 sin 和 cos
        # embedding 形状: [Batch, dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # 如果维度是奇数(很少见)，补一个零
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, t):
        # 1. 先进行数学编码 (无参数)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        # 2. 再过 MLP (有参数)
        t_emb = self.mlp(t_freq)
        return t_emb