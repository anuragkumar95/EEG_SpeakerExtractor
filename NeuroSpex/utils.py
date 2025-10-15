import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_collate_fn(batch):
    a_mix, a_tgt, ref_tgt = batch[0]
    a_mix = torch.tensor(a_mix).unsqueeze(1)
    a_tgt = torch.tensor(a_tgt).unsqueeze(1) 
    ref_tgt = torch.tensor(ref_tgt).permute(0, 2, 1) 
    return a_mix, a_tgt, ref_tgt

# Implementation taken from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/time.py
class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Note that this returns the negative of the SI-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):

        if len(target.shape) == 3:
            target = target.squeeze(1)
        target = target[:, :input.shape[-1]]
        
        assert input.shape == target.shape, f"Input and target must have the same shape, got {input.shape} and {target.shape}"

        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        if self.reduction == "mean":
            losses = losses.mean()
        else:
            losses = losses.sum()
        return -losses