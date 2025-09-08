"""
Author: Anurag Kumar
Created on: 2025-09-07
"""

import torch
import torch.nn as nn
from eeg_modules import EEGEncoder
from speech_modules import SpeechEncoder, SpeechDecoder
from speaker_extractor import SpeakerExtractor


class NeuroSpex(nn.Module):
    def __init__(self, eeg_channels, speech_out_channels):
        super(NeuroSpex, self).__init__()
        self.eeg_encoder = EEGEncoder(input_ch=eeg_channels, num_heads=2, n_adcblocks=4)
        self.speech_encoder = SpeechEncoder(input_ch=1, output_ch=speech_out_channels, kernel_size=20, stride=10, padding=0)
        self.speech_decoder = SpeechDecoder(input_ch=256, output_ch=1, kernel_size=20, stride=10, padding=0)
        self.speaker_extractor = SpeakerExtractor(eeg_ch=64, speech_ch=256)

    def forward(self, speech, eeg):
        """
        speech : (batch, 1, T_x)
        eeg : (batch, 64, T_y)
        """
        # Encode
        eeg_emb = self.eeg_encoder(eeg)
        speech_emb = self.speech_encoder(speech).permute(0, 2, 1)
        
        # Get speaker mask
        mask = self.speaker_extractor(speech_emb, eeg_emb)
        spk_out = (mask * speech_emb).permute(0, 2, 1)

        # Decode
        out = self.speech_decoder(spk_out)
        return out