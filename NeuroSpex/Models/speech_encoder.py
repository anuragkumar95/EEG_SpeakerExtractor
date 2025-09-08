import torch.nn as nn
import torch.nn.functional as F

class SpeechEncoder(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, padding=0):
        super(SpeechEncoder, self).__init__()
        self.conv1d = nn.Conv1d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = F.pad(x, (0, 10))
        outputs = F.relu(self.conv1d(x))
        return outputs