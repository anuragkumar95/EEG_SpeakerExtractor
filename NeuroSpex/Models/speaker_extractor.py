"""
Author: Anurag Kumar
Created on: 2025-09-07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convtasnet_separator import Separator

# Parts of this code are taken from https://github.com/JusperLee/Conv-TasNet/blob/master/Conv_TasNet_Pytorch/Conv_TasNet.py

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm not in ['gln', 'cln', 'bn']:
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input. 
       It is also known as a fractionally-strided convolution 
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.Sc_conv(c)
        return x+c

def Sequential_block(num_blocks, **block_kwargs):
      '''
          Sequential 1-D Conv Block
          input:
                num_block: how many blocks in every repeats
                **block_kwargs: parameters of Conv1D_Block
      '''
      Conv1D_Block_lists = [Conv1D_Block(
          **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

      return nn.Sequential(*Conv1D_Block_lists)

def Sequential_repeat(num_repeats, num_blocks, **block_kwargs):
      '''
          Sequential repeats
          input:
                num_repeats: Number of repeats
                num_blocks: Number of block in every repeats
                **block_kwargs: parameters of Conv1D_Block
      '''
      repeats_lists = [Sequential_block(
          num_blocks, **block_kwargs) for i in range(num_repeats)]
      return nn.Sequential(*repeats_lists)


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
        attention_weights = F.softmax(scores, dim=-1)

        # 6. Apply attention to values
        # Context shape: [B, num_heads, T_q, head_dim]
        context = torch.einsum('bnqk,bnke->bnqe', attention_weights, v)
        
        # 7. Concatenate heads and project back
        # Reshape context to [B, T_q, embed_dim]
        context = context.transpose(1, 2).contiguous().view(B, T_q, E)
        output = self.out_proj(context)

        return output#, attention_weights

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels, n_ca_heads=1):
        super(CrossAttnBlock, self).__init__()
        self.cross_attn = CrossAttention(embed_dim=in_channels, num_heads=n_ca_heads)
        self.layer_norm = nn.LayerNorm(in_channels)
    
    def forward(self, speech_emb, eeg_emb):
        """
        speech_emb : (batch, T_x, 64)
        eeg_emb : (batch, T_x, 64), interpolated.
        """
        #attn_out, attn_w = self.cross_attn(query=eeg_emb, key=speech_emb, value=speech_emb)
        attn_out = self.cross_attn(query=eeg_emb, key=speech_emb, value=speech_emb)
        attn_out = self.layer_norm(eeg_emb + attn_out)
        return attn_out

        
class CrossAttnTCNBlock(nn.Module):
    def __init__(self, in_channels, tcn_channels=512, n_ca_heads=1, n_tcn_layers=3, n_tcn_depth=8, tcn_kernel_size=3, causal=False):
        super(CrossAttnTCNBlock, self).__init__()
        self.cross_attn = CrossAttnBlock(in_channels=in_channels, n_ca_heads=n_ca_heads)
        # self.tcn = Sequential_repeat(
        #     num_repeats=n_tcn_layers, 
        #     num_blocks=n_tcn_depth, 
        #     in_channels=in_channels, 
        #     out_channels=in_channels, 
        #     kernel_size=tcn_kernel_size, 
        #     norm="gln", 
        #     causal=causal)
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
        tcn_out = self.tcn(attn_out.permute(0, 2, 1))
        return tcn_out

class SpeakerExtractor(nn.Module):
    def __init__(self, eeg_ch, speech_ch, n_ca_blocks=4, n_ca_heads=1, n_tcn_layers=3, n_tcn_depth=8, tcn_kernel_size=3, tcn_channels=512, causal=False):
        super(SpeakerExtractor, self).__init__()
        self.layer_norm = nn.LayerNorm(speech_ch)
        #self.dropout = nn.Dropout(p=0.1)
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
        speech_emb = self.conv1(speech_emb)
       
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