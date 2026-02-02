import torch, math
import torch.nn as nn
import einx


class PE(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros((max_seq_len, d_model))

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # pe_{pos,2i} = sin(\frac{pos}{10000^{2i/d_model}})
                pe[pos, i] = math.sin(pos / (1e4 ** ((2 * i) / d_model)))
                # pe_{pos,2i+1} = cos(\frac{pos}{10000^{2i/d_model}})
                pe[pos, i + 1] = math.cos(pos / (1e4 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, : x.size(1), :]

        return x


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 512, 768
    x = torch.randn((batch_size, seq_len, d_model))

    pe = PE(d_model, max_seq_len=1024)
    out = pe(x)
    print(f": {out.shape}")
