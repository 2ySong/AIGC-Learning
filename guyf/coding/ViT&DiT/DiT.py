import torch
import torch.nn as nn
import numpy as np
import math

from einops import rearrange

from vit import Attention


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    def timestep_embedding(self,t, dim, max_period=10000):
        freqs=torch.exp(
            -math.log(max_period)*torch.arange(0,dim//2,dtype=torch.float32)
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def forward(self, x):
        t_freq=self.timestep_embedding(x,self.frequency_embedding_size)
        t_emb=self.mlp(t_freq)
        return t_emb
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dim_head=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, heads=num_heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = FFN(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
class FFN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_size, ):
        super().__init__()
        self.net=nn.Conv2d(in_channels,hidden_size,kernel_size=patch_size[0],stride=patch_size[1])
    def forward(self,x):
        return self.net(x).flatten(2).transpose(1,2)

def pair(t):
    '''
    return tuple of imgsize
    '''
    return t if isinstance(t, tuple) else (t, t)
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        dim_head=64,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size=pair(input_size)
        self.patch_size = pair(patch_size)
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = (self.input_size[0]//self.patch_size[0])*(self.input_size[1]//self.patch_size[1])
        #替换
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        #
        self.blocks = nn.ModuleList([
                    DiTBlock(hidden_size, num_heads,dim_head=dim_head, mlp_ratio=mlp_ratio) for _ in range(depth)
                ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
    def unpatchify(self, x):
        """
        x: (B, H*W, patch_h*patch_w * C)
        out: (B, C, H, W)
        """
        return rearrange(x,'b (h w) (p_h p_w c)-> b c (h p_h) (w p_w)',
                   p_h=self.patch_size[0], p_w=self.patch_size[1],h=self.input_size[0] // self.patch_size[0])

    def forward(self, x, t,y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y,train=False)
        c = t + y
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. 定义模型参数（使用轻量化参数，避免显存溢出）
    model_config = {
        "input_size": 32,  # 输入图像尺寸 (32x32)
        "patch_size": (2, 2),  # Patch大小 2x2
        "in_channels": 4,  # 输入通道数
        "hidden_size": 128,  # 隐藏层维度
        "depth": 2,  # Transformer块数量
        "num_heads": 4,  # 注意力头数
        'dim_head':64,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 10,
        "learn_sigma": False
    }

    # 3. 初始化DiT模型
    model = DiT(**model_config)
    model.eval()  # 推理模式
    # 4. 生成随机输入矩阵（模拟真实输入）
    batch_size = 2  # 批次大小
    # 4.1 图像输入 x: [batch_size, in_channels, H, W] → [2,4,32,32]
    x = torch.randn(batch_size, model_config["in_channels"], model_config["input_size"], model_config["input_size"])
    # 4.2 时间步 t: [batch_size] → [2]（扩散模型的时间步，范围0~1000）
    t = torch.randint(0, 1000, (batch_size,))
    # 4.3 类别标签 y: [batch_size] → [2]（随机类别，0~9）
    y = torch.randint(0, model_config["num_classes"], (batch_size,))

    # 5. 打印输入矩阵信息（维度+部分数值）
    print("=== 输入矩阵信息 ===")
    print(f"1. 图像输入 x 维度: {x.shape}")
    print(f"2. 时间步 t 维度: {t.shape}, 数值: {t.numpy()}")
    print(f"3. 类别标签 y 维度: {y.shape}, 数值: {y.numpy()}")
    print("-" * 50)

    # 6. 前向传播（禁用梯度计算，节省显存）
    with torch.no_grad():
        output = model(x, t, y)

    # 7. 打印输出矩阵信息（维度+部分数值）
    print("=== 输出矩阵信息 ===")
    print(f"1. 模型输出 output 维度: {output.shape}")
    print(f"   输出通道数: {output.shape[1]} (learn_sigma=True → 4×2=8)")
    print("-" * 50)