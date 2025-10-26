"""
Author: Anurag Kumar
Created on: 2025-09-07
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2).contiguous() # (batch, seq_len, embed_dim)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.out_proj = nn.Linear(num_heads*embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project embeddings to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Calculate attention scores (scaled dot-product attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Apply attention mask

        attn_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attn_weights, v)

        # Concatenate heads and project to output
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attended_values)
        return output

class ADCBlock(nn.Module):
    def __init__(self, input_ch, num_heads=1, kernel_size=3, dropout=0.1):
        super(ADCBlock, self).__init__()
        self.pos_enc = PositionalEncoding(d_model=input_ch)
        self.mha = MultiHeadAttention(embed_dim=input_ch, num_heads=num_heads)
        self.depth_conv = nn.Conv1d(input_ch, input_ch, kernel_size=kernel_size, stride=1, padding='same', groups=input_ch)
        self.layer_norm = nn.LayerNorm(input_ch)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MHA step
        x = self.pos_enc(x.permute(1, 0, 2))
        mha_out = self.mha(x)
        x = self.layer_norm(x + mha_out)
   
        # Depth Conv step
        x = x.permute(0, 2, 1)  
        x = x + self.depth_conv(x)
        x = self.layer_norm(x.permute(0, 2, 1))
        return x
            
class EEGEncoder(nn.Module):
    def __init__(self, input_ch, num_heads=2, n_adcblocks=1, kernel_size=10, dropout=0.1):
        super(EEGEncoder, self).__init__()
        self.pre_conv = nn.Conv1d(input_ch, input_ch, kernel_size=3, stride=1, padding='same')
        self.ADCBlocks = nn.ModuleList()
        for _ in range(n_adcblocks):
            self.ADCBlocks.append(ADCBlock(input_ch=input_ch, num_heads=num_heads, kernel_size=kernel_size, dropout=dropout))

    def forward(self, x):
        x = F.relu(self.pre_conv(x).permute(0, 2, 1))
        for adc_block in self.ADCBlocks:
            x = adc_block(x)
        return x