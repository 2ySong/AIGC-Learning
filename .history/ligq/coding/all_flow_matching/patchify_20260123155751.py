import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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

print("-" * 30)
print("真实数据测试结果:")
print(f"输入图片形状: {images.shape} (一张 32x32 的 RGB 图)")
print("-" * 30)
print(f"Patch Size 设置为: {model.patch_size} x {model.patch_size}")
print(f"计算出的 Patch 数量: {model.n_patches} (即序列长度)")
print(f"每个 Patch 被映射到的维度 (Embed Dim): {128}")
print("-" * 30)
print(f"模型输出形状: {output_sequence.shape}")
print("解读: [Batch Size=1, 序列长度=16, 每个token的维度=128]")
print("-" * 30)

# 验证一下输出里的数据是不是真的不一样的 (防止模型没初始化好全输出0)
print("打印第一个 Patch 对应的向量的前5个数值 (验证有数据流动):")
print(output_sequence[0, 0, :5].detach().numpy())