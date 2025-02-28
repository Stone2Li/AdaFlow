import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feedforward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class Flow_Matching(nn.Module):
    def __init__(self, seq_len = 1000, input_dim = 1, hidden_dim = 128, num_heads = 8, num_layers = 6, output_dim = 1000):
        super(Flow_Matching, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 嵌入 t 和 label
        self.label_embedding = nn.Linear(seq_len, hidden_dim)        
        # 定义多个Transformer层
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])

        # 输出层
        self.fc_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim//2, input_dim))

    def time_emb(self,t,dim=1000):
        t = t * 1000
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2+dim % 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)[:,:dim]
    
    def label_emb(self,label,dim=1000):
        y = label * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2+dim % 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)[:,:dim]
    
    def forward(self, x, t, label, seq_len):
        # 输入：x.shape = [batch_size, seq_len, input_dim], t 和 label 是额外的输入

        # 输入嵌入
        x = self.embedding(x)  # 转换为隐藏维度 [batch_size, seq_len, hidden_dim]
        
        # 嵌入 t 和 label
        t_emb = self.time_emb(t, dim=seq_len)  # [batch_size, hidden_dim]
        label_emb = self.label_emb(label, dim=seq_len)  # [batch_size, hidden_dim]
        emb = t_emb + label_emb
        emb = self.label_embedding(emb)
        # 将 t 和 label 嵌入加到 x 上
        x = x + emb.unsqueeze(1)  # 广播到 seq_len

        # 转置为 [seq_len, batch_size, hidden_dim] 以适应多头自注意力机制
        x = x.permute(1, 0, 2)

        # 经过多个Transformer层
        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)
        # 输出层

        output = self.fc_out(x)  # [batch_size, output_dim]
        
        return output


if __name__ == '__main__':
    # 示例参数
    input_dim = 1  # 每个位点甲基化值1维
    hidden_dim = 64  # 隐藏维度
    num_heads = 8  # 多头注意力头数
    num_layers = 6  # Transformer层数
    output_dim = 1000  # 预测迁移速度（1维）
    seq_len = 1000
    # 模型初始化
    model = Flow_Matching(seq_len, input_dim, hidden_dim, num_heads, num_layers, output_dim)

    # 测试模型
    sample_input = torch.randn(60, 1000, input_dim)  # (batch_size, seq_len, input_dim)
    t_input = torch.rand(60)   # (batch_size,) 假设有60个样本，每个样本有一个时间戳
    label_input = torch.randint(0, 1, (60,))  # (batch_size,) 假设标签为2类

    output = model(sample_input, t_input, label_input)  # 输出: (batch_size, output_dim)
    print(output.shape)  # 预期是 (60, 1)，即每个样本的迁移速度预测
    