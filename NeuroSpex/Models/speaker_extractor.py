"""
Author: Anurag Kumar
Created on: 2025-09-07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convtasnet_separator import TCN_Stack

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
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

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum('bnqe,bnke->bnqk', q, k) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.einsum('bnqk,bnke->bnqe', attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T_q, E)
        output = self.out_proj(context)

        return output#, attention_weights

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels, n_ca_heads=1, dropout=0.1):
        super(CrossAttnBlock, self).__init__()
        self.cross_attn = CrossAttention(embed_dim=in_channels, num_heads=n_ca_heads)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, speech_emb, eeg_emb, speech_query=False):
        """
        speech_emb : (batch, T_x, 64)
        eeg_emb : (batch, T_x, 64), interpolated.
        """
        if not speech_query:
            attn_out = self.cross_attn(query=eeg_emb, key=speech_emb, value=speech_emb)
            attn_inp = eeg_emb
        else:
            attn_out = self.cross_attn(query=speech_emb, key=eeg_emb, value=eeg_emb)
            attn_inp = speech_emb
            
        attn_out = self.dropout(attn_out)
        attn_out = attn_out + attn_inp
        attn_out = self.layer_norm(attn_out)
        return attn_out

class CrossAttnTCNBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            tcn_channels=512, 
            n_ca_heads=1, 
            n_tcn_layers=3, 
            n_tcn_depth=8, 
            tcn_kernel_size=3, 
            dropout=0, 
            causal=False
        ):
        super(CrossAttnTCNBlock, self).__init__()
        self.cross_attn = CrossAttnBlock(
            in_channels=in_channels, 
            n_ca_heads=n_ca_heads, 
            dropout=dropout
        )
        self.tcn = TCN_Stack( 
            N=in_channels, 
            B=in_channels,
            H=tcn_channels, 
            P=tcn_kernel_size, 
            X=n_tcn_depth, 
            R=n_tcn_layers, 
            causal=causal
        )
   
    def forward(self, speech_emb, eeg_emb, speech_query=False):
        """
        speech_emb : (batch, T_x, 64)
        eeg_emb : (batch, T_x, 64), interpolated.
        """
        attn_out = self.cross_attn(speech_emb, eeg_emb, speech_query)
        if not speech_query:
            attn_out = speech_emb + attn_out
        tcn_out = self.tcn(attn_out)
        return tcn_out

class SpeakerExtractor(nn.Module):
    def __init__(
        self, 
        eeg_ch, 
        speech_ch, 
        n_ca_blocks=4, 
        n_ca_heads=1, 
        n_tcn_layers=3, 
        n_tcn_depth=8, 
        tcn_kernel_size=3, 
        tcn_channels=512, 
        dropout=0, 
        causal=False
    ):
        super(SpeakerExtractor, self).__init__()
        self.layer_norm = nn.LayerNorm(speech_ch)
        self.conv_in = nn.Conv1d(speech_ch, eeg_ch, kernel_size=1, stride=1, padding=0)
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
        embedding = embedding.transpose(1, 2) 
        # Perform linear interpolation
        interpolated_embedding = F.interpolate(
            embedding,
            size=tgt_seq_len,
            mode='linear',
            align_corners=False # Set to False for non-boundary-aligned interpolation
        )
        return interpolated_embedding

    def forward(self, speech_emb, eeg_emb, speech_query=False):
        """
        speech_emb : (batch, T_x, 256)
        eeg_emb : (batch, T_y, 64).
        """
        assert speech_emb.shape[-1] == 256, f"Speech EMB:{speech_emb.shape}"
        assert eeg_emb.shape[-1] == 64, f"Speech EMB:{eeg_emb.shape}"

        speech_emb = self.layer_norm(speech_emb).permute(0, 2, 1)
        speech_emb = self.conv_in(speech_emb) # Bottleneck
       
        #Interpolate eeg embedding
        if speech_query:
            eeg_emb = eeg_emb.permute(0, 2, 1) # Interpolation is not required when using speech query
        else:
            eeg_emb = self.interpolate(eeg_emb, speech_emb.shape[-1])
        
        #Convert both embeddings in shape (B , seq_len , channels)
        speech_emb = speech_emb.permute(0, 2, 1)
        eeg_emb = eeg_emb.permute(0, 2, 1)

        for k, ca_block in enumerate(self.ca_blocks):
            attn_out = ca_block(speech_emb, eeg_emb, speech_query)
            if not speech_query:
                eeg_emb = attn_out.permute(0, 2, 1)
            else:
                speech_emb = attn_out.permute(0, 2, 1)

        if not speech_query:
            mask = F.relu(self.conv_out(eeg_emb.permute(0, 2, 1)))
        else:
            mask = F.relu(self.conv_out(speech_emb.permute(0, 2, 1)))
        
        return mask.permute(0, 2, 1)