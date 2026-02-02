import torch, math
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        # 计算频率：theta_i = 10000^(-2i/d) for i in [0, d/2)
        theta = 1.0 / (1e4 ** (torch.arange(0, d_model, 2).float() / d_model))

        # 位置索引
        pos = torch.arange(max_seq_len).float()
        # 计算频率矩阵：m * theta
        freqs = torch.outer(pos, theta)  # [max_seq_len, d_model//2]
        
        # 为了适配完整的d_model维度，需要重复频率（每对维度使用相同的频率）
        freqs = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, d_model]

        self.register_buffer(
            "cos", torch.cos(freqs).unsqueeze(0).unsqueeze(2)
        )  # [1, max_seq_len, 1, d_model]
        self.register_buffer(
            "sin", torch.sin(freqs).unsqueeze(0).unsqueeze(2)
        )  # [1, max_seq_len, 1, d_model]

    def forward(self, q, k):
        seq_len = q.size(1)
        cos = self.cos[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]
        sin = self.sin[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]

        # 旋转变换：将向量分成前后两半 [x0, x1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
        # 旋转后：[x0*cos - x_{d/2}*sin, ..., x_{d/2}*cos + x0*sin, ...]
        d = q.shape[-1]
        q_rot = torch.cat([-q[..., d//2:], q[..., :d//2]], dim=-1)
        k_rot = torch.cat([-k[..., d//2:], k[..., :d//2]], dim=-1)

        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin

        return q_out, k_out

if __name__ == "__main__":
    batch_size, seq_len, num_heads, d_model = 2, 512, 8, 64
    q = torch.randn((batch_size, seq_len, num_heads, d_model))
    k = torch.randn((batch_size, seq_len, num_heads, d_model))

    rope = RoPE(d_model=d_model, max_seq_len=1024)
    q_out, k_out = rope(q, k)
    print(f"q_out: {q_out.shape}, k_out: {k_out.shape}")