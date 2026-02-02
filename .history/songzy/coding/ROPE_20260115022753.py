import torch, math
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        theta = 1.0 / (1e4 ** (torch.arange(0, d_model, 2) / d_model))

        pos = torch.arange(max_seq_len)
        freqs = torch.outer(pos, theta).repeat(1, 2)  # [max_seq_len, d_model]

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

        q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)

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