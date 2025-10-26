"""
Author: Anurag Kumar
Created on: 2025-09-07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convtasnet_separator import Separator


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

        # Final linear layer to combine head outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: The sequence to attend from (e.g., shape: [B, T_q, E])
            key: The sequence to attend to (e.g., shape: [B, T_k, E])
            value: The values of the sequence to attend to (e.g., shape: [B, T_k, E])
            mask: An optional attention mask to handle padding (e.g., shape: [B, T_q, T_k])
        """
        B, T_q, E = query.shape
        _, T_k, _ = key.shape

        # 1. Project Q, K, V
        # All shapes will be [B, T, E]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape and split into heads
        # Shapes become [B, num_heads, T, head_dim]
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        # Score shape: [B, num_heads, T_q, T_k]
        scores = torch.einsum('bnqe,bnke->bnqk', q, k) / (self.head_dim ** 0.5)

        # 4. Apply mask if provided
        if mask is not None:
            # Mask has shape [B, T_q, T_k], so we unsqueeze to match scores
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # 5. Softmax to get attention weights
        attention_weights = self.dropout(F.softmax(scores, dim=-1))

        # 6. Apply attention to values
        # Context shape: [B, num_heads, T_q, head_dim]
        context = torch.einsum('bnqk,bnke->bnqe', attention_weights, v)
        
        # 7. Concatenate heads and project back
        # Reshape context to [B, T_q, embed_dim]
        context = context.transpose(1, 2).contiguous().view(B, T_q, E)
        output = self.out_proj(context)

        return output#, attention_weights

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels, n_ca_heads=1, dropout=0.1):
        super(CrossAttnBlock, self).__init__()
        self.cross_attn = CrossAttention(embed_dim=in_channels, num_heads=n_ca_heads)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, speech_emb, eeg_emb):
        """
        speech_emb : (batch, T_x, 64)
        eeg_emb : (batch, T_x, 64), interpolated.
        """
        #attn_out, attn_w = self.cross_attn(query=eeg_emb, key=speech_emb, value=speech_emb)
        attn_out = self.cross_attn(query=eeg_emb, key=speech_emb, value=speech_emb)
        attn_out = self.dropout(attn_out)
        attn_out = eeg_emb + attn_out
        attn_out = self.layer_norm(attn_out)
        return attn_out


class CrossAttnTCNBlock(nn.Module):
    def __init__(self, in_channels, tcn_channels=512, n_ca_heads=1, n_tcn_layers=3, n_tcn_depth=8, tcn_kernel_size=3, dropout=0, causal=False):
        super(CrossAttnTCNBlock, self).__init__()
        self.cross_attn = CrossAttnBlock(in_channels=in_channels, n_ca_heads=n_ca_heads, dropout=dropout)
        self.tcn = Separator( 
            N=in_channels, 
            B=in_channels, #Bottleneck channels
            H=tcn_channels, 
            P=tcn_kernel_size, 
            X=n_tcn_depth, 
            R=n_tcn_layers, 
            causal=causal)
   
    
    def forward(self, speech_emb, eeg_emb):
        """
        speech_emb : (batch, T_x, 64)
        eeg_emb : (batch, T_x, 64), interpolated.
        """
        attn_out = self.cross_attn(speech_emb, eeg_emb)
        attn_out = speech_emb + attn_out
        attn_out = attn_out.permute(0, 2, 1)  # (B, C, T)
        tcn_out = self.tcn(attn_out.permute(0, 2, 1))
        return tcn_out


class SpeakerExtractor(nn.Module):
    def __init__(self, eeg_ch, speech_ch, n_ca_blocks=4, n_ca_heads=1, n_tcn_layers=3, n_tcn_depth=8, tcn_kernel_size=3, tcn_channels=512, dropout=0, causal=False):
        super(SpeakerExtractor, self).__init__()
        self.layer_norm = nn.LayerNorm(speech_ch)
        self.conv1 = nn.Conv1d(speech_ch, eeg_ch, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv1d(eeg_ch, speech_ch, kernel_size=1, stride=1, padding=0)
        self.ca_blocks = nn.ModuleList([
            CrossAttnTCNBlock(
                in_channels=eeg_ch, 
                tcn_channels=tcn_channels,
                n_ca_heads=n_ca_heads, 
                n_tcn_layers=n_tcn_layers,
                n_tcn_depth=n_tcn_depth,
                tcn_kernel_size=tcn_kernel_size,
                dropout=dropout,
                causal=causal) for _ in range(n_ca_blocks)
        ])

    def interpolate(self, embedding, tgt_seq_len):
        # To use interpolate, the input needs to have a specific shape: (N, C, L)
        # where N=batch, C=channels, L=length.
        # So, we swap the sequence length and embedding dimensions.
        embedding = embedding.transpose(1, 2)  # Shape becomes: (1, 512, 10)

        # Perform linear interpolation
        interpolated_embedding = F.interpolate(
            embedding,
            size=tgt_seq_len,
            mode='linear',
            align_corners=False # Set to False for non-boundary-aligned interpolation
        )

        return interpolated_embedding

    def forward(self, speech_emb, eeg_emb):
        """
        speech_emb : (batch, T_x, 256)
        eeg_emb : (batch, T_y, 64).
        """
        assert speech_emb.shape[-1] == 256, f"Speech EMB:{speech_emb.shape}"
        assert eeg_emb.shape[-1] == 64, f"Speech EMB:{eeg_emb.shape}"

        speech_emb = self.layer_norm(speech_emb).permute(0, 2, 1)
        speech_emb = self.conv1(speech_emb) # Bottleneck
       
        #Interpolate eeg embedding
        eeg_emb = self.interpolate(eeg_emb, speech_emb.shape[-1])
        #eeg_emb = eeg.emb.permute(0, 2, 1) #Emulate interpolation shape change

        #Convert both embeddings in shape (B , seq_len , channels)
        speech_emb = speech_emb.permute(0, 2, 1)
        eeg_emb = eeg_emb.permute(0, 2, 1)
        
        for k, ca_block in enumerate(self.ca_blocks):
            attn_out = ca_block(speech_emb, eeg_emb)
            eeg_emb = attn_out.permute(0, 2, 1)

        mask = F.relu(self.conv_out(eeg_emb.permute(0, 2, 1)))
        return mask.permute(0, 2, 1)