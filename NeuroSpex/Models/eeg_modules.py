"""
Author: Anurag Kumar
Created on: 2025-09-07
"""
import math
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len):
        # x shape: (batch_size, num_heads, seq_len, head_dim) or similar
        # We need to create position-based angles
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # Duplicate to match dimensions

        # Apply rotation
        # The rotation needs to be applied to pairs of dimensions
        # For simplicity, this example shows applying it to the entire vector
        # A proper implementation would split x into pairs and rotate each pair
        # Example here assumes x is already structured for pairwise rotation or has same dim as emb
        rotated_x = x * emb.cos() + self.rotate_half(x) * emb.sin()
        return rotated_x

    def rotate_half(self, x):
        # Splits the vector into two halves and rotates the second half
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:]
        return torch.cat((-x2, x1), dim=-1)

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads*embed_dim)
        self.out_proj = nn.Linear(num_heads*embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim, base=10000) # RoPE applied per head

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

        # Apply RoPE to queries and keys
        q_rotated = self.rope(q, seq_len)
        k_rotated = self.rope(k, seq_len)

        # Transpose for attention calculation (batch, num_heads, seq_len, head_dim)
        q_rotated = q_rotated.transpose(1, 2)
        k_rotated = k_rotated.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores (scaled dot-product attention)
        scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Apply attention mask

        attn_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attn_weights, v)

        # Concatenate heads and project to output
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attended_values)
        return output


class PreConv(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=3):
        super(PreConv, self).__init__()
        self.conv1d = nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, stride=1, padding='same')
    
    def forward(self, x):
        outputs = F.relu(self.conv1d(x)).permute(0, 2, 1)
        return outputs

class ADCBlock(nn.Module):
    def __init__(self, input_ch, num_heads, kernel_size=3):
        super(ADCBlock, self).__init__()
        self.mha = MultiHeadAttentionWithRoPE(embed_dim=input_ch, num_heads=num_heads)
        self.depth_conv = nn.Conv1d(input_ch, input_ch, kernel_size=kernel_size, stride=1, padding='same', groups=input_ch)
        self.layer_norm = nn.LayerNorm(input_ch)

    def forward(self, x):
        # MHA step
        x = self.layer_norm(x + self.mha(x))
        # Depth Conv step
        x = x.permute(0, 2, 1)  
        x = x + self.depth_conv(x)
        x = self.layer_norm(x.permute(0, 2, 1))
        return x
            
class EEGEncoder(nn.Module):
    def __init__(self, input_ch, num_heads=3, n_adcblocks=1):
        super(EEGEncoder, self).__init__()
        self.pre_conv = PreConv(input_ch=input_ch, output_ch=input_ch, kernel_size=10)
        self.ADCBlocks = nn.ModuleList()
        for _ in range(n_adcblocks):
            self.ADCBlocks.append(ADCBlock(input_ch=input_ch, num_heads=num_heads, kernel_size=10))

    def forward(self, x):
        x = self.pre_conv(x)
        for adc_block in self.ADCBlocks:
            x = adc_block(x)
        return x