import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
import math
import copy

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Separator(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''
    def __init__(self, N=64, B=64, H=512, P=3, X=8, R=3, causal=False):
        super(Separator, self).__init__()

        if causal:
            self.layer_norm = cLN(N)
        else:
            self.layer_norm = ChannelWiseLayerNorm(N)
        #self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1)

        self.tcn = _clones(TCN_block(X,P,B,H,causal), R)
        #self.mask_conv1x1 = nn.Conv1d(B, N, 1)


    def forward(self, x):
       # K = x.size()[-1]

        x = self.layer_norm(x)
        #x = self.bottleneck_conv1x1(x)

        # tcn blocks
        for i in range(len(self.tcn)):
            x = self.tcn[i](x)

        #x = self.mask_conv1x1(x)
        x = F.relu(x)
        return x

class TCN_block(nn.Module):
    def __init__(self, X, P, B, H, causal):
        super(TCN_block, self).__init__()
        tcn_blocks = []
        for x in range(X):
            tcn_blocks += [Conv1DBlock(B, H, P, dilation = 2**x, causal=causal)]
        self.tcn = nn.Sequential(*tcn_blocks)

    def forward(self, x):
        x = self.tcn(x)
        return x

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        if causal:
            self.lnorm1 = cLN(conv_channels)
        else:
            self.lnorm1 = ChannelWiseLayerNorm(conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation)
        self.prelu2 = nn.PReLU()
        if causal:
            self.lnorm2 = cLN(conv_channels)
        else:
            self.lnorm2 = ChannelWiseLayerNorm(conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

