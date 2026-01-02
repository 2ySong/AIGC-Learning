import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        """
        多头注意力机制的实现。
        Args:
            hidden_size (int): 输入特征的维度，也即 hidden_state 的最后一维。
            num_heads (int): 注意力头的数量。
            dropout (float): dropout 的概率，默认为 0.0。
        """
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度

        # 定义线性变换层，用于生成 Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # 输出线性层
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):

        """
        前向传播函数。
        Args:
            hidden_state (torch.Tensor): 输入的 hidden_state，形状为 [batch_size, seq_len, hidden_size]。
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽某些位置，形状为 [batch_size, seq_len]。默认为 None。
        Returns:
            torch.Tensor: 注意力输出，形状为 [batch_size, seq_len, hidden_size]。

        """
        batch_size, seq_len, _ = hidden_state.size()

        # 1. 通过线性层得到 Q, K, V
        query = self.query(hidden_state)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_state)      # [batch_size, seq_len, hidden_size]
        value = self.value(hidden_state)    # [batch_size, seq_len, hidden_size]

        # 2. 将 Q, K, V 拆分成多头
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # [batch_size, num_heads, seq_len, head_dim]

        # 3. 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]

        # 应用 attention mask
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))

        attention_weights = torch.softmax(attention_weights, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = self.dropout(attention_weights)

        # 4. 计算上下文向量
        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, head_dim]

        # 5. 将多头合并
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        # 6. 通过输出线性层
        output = self.out_projection(context)  # [batch_size, seq_len, hidden_size]
        return output
if __name__ == '__main__':
    # 示例
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8

    # 创建一个 MHA 实例
    mha = MultiHeadAttention(hidden_size, num_heads)

    # 创建一个随机的 hidden_state
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    # 创建一个 attention mask (可选)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 5:] = 0  # 屏蔽掉每个 batch 中 seq_len 的后 5 个位置

    # 通过 MHA 层
    output = mha(hidden_state, attention_mask)

    # 打印输出形状
    print("输出形状:", output.shape)  # torch.Size([2, 10, 256])
