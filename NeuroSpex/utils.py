import torch
import torch.nn as nn
import torch.nn.functional as F


class SiSDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SiSDRLoss, self).__init__()
        self.eps = eps

    def forward(self, est, target):
        """
        est: (batch, T)
        target: (batch, T)
        """
        # Zero-mean normalization
        est = est - torch.mean(est, dim=1, keepdim=True)
        target = target - torch.mean(target, dim=1, keepdim=True)

        # Compute scaling factor
        alpha = torch.sum(est * target, dim=1, keepdim=True) / (torch.sum(target ** 2, dim=1, keepdim=True) + self.eps)

        # Project estimated signal onto target
        proj = alpha * target

        # Compute noise
        noise = est - proj

        # Compute SI-SDR
        si_sdr = 10 * torch.log10((torch.sum(proj ** 2, dim=1) + self.eps) / (torch.sum(noise ** 2, dim=1) + self.eps))

        # Return negative SI-SDR as loss
        return -torch.mean(si_sdr)