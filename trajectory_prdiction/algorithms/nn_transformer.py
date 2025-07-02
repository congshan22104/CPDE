# model.py

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    标准的正弦/余弦位置编码（固定，不可训练），
    参考原论文 “Attention Is All You Need”
    """
    def __init__(self, d_model: int, max_len: int = 500):
        """
        ---------- 
        d_model : int
            Transformer 特征维度大小
        max_len : int
            序列最大长度，上限，一般取 L_in
        """
        super().__init__()
        # pe 最终形状 (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor, shape (B, seq_len, d_model)
        返回 x + pe[:, :seq_len, :]
        """
        seq_len = x.size(1)
        # 注意：pe 的第一个维度是 1，对应 batch，所以可以直接广播
        x = x + self.pe[:, :seq_len, :]
        return x


class Transformer(nn.Module):
    """
    用 Transformer Encoder 结构，把过去 L_in 步的 (x,y,z) 序列映射到未来 L_out 步 (x,y,z)。

    具体流程：
      1. 输入部分：Linear 将 3 维坐标投射到 embed_dim
      2. 位置编码：加上 PositionalEncoding
      3. TransformerEncoder 层叠
      4. 从 Encoder 输出里取最后一个时刻的隐藏向量，接一个 MLP 映射到 (L_out * 3)
      5. reshape → (L_out, 3)
    """
    def __init__(self,
                 input_dim: int = 3,
                 embed_dim: int = 128,
                 n_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 L_in: int = 10,
                 L_out: int = 5):
        """
        ---------- 
        input_dim : int
            原始输入维度，本例中是 3 (x,y,z)
        embed_dim : int
            Transformer 内部特征维度
        n_heads : int
            多头注意力头数
        num_layers : int
            TransformerEncoder 层数
        dropout : float
            Dropout 比例
        L_in : int
            历史序列长度
        L_out : int
            需要预测的未来步数
        """
        super().__init__()
        self.L_in = L_in
        self.L_out = L_out

        # 1) 将 3 维映射到 embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 2) 位置编码
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=L_in)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='relu',
            batch_first=True   # 接受输入 (B, Seq, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) 从最后一个隐藏向量映射到 L_out * 3
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, L_out * input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        ----------
        x : Tensor, shape (B, L_in, 3)
        返回
        -------
        pred : Tensor, shape (B, L_out, 3)
            预测的未来位置序列
        """
        # x_emb: (B, L_in, embed_dim)
        x_emb = self.input_proj(x)
        # 加位置编码
        x_emb = self.pos_encoder(x_emb)
        # Transformer Encoder
        enc_out = self.transformer_encoder(x_emb)  # (B, L_in, embed_dim)
        # 取最后一个时刻的输出
        last_hidden = enc_out[:, -1, :]            # (B, embed_dim)
        # 预测头：先映射到 L_out * 3，再 reshape
        pred = self.pred_head(last_hidden)         # (B, L_out * 3)
        pred = pred.view(-1, self.L_out, 3)        # (B, L_out, 3)
        return pred
