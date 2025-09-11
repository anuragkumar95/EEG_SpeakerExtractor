"""
Author: Anurag Kumar
Created on: 2025-09-07
"""

import torch
import torch.nn as nn
from .eeg_modules import EEGEncoder
from .speech_modules import SpeechEncoder, SpeechDecoder
from .speaker_extractor import SpeakerExtractor


class NeuroSpex(nn.Module):
    def __init__(self, 
        speech_encoder_params,
        eeg_encoder_params,
        speech_decoder_params,
        spk_ext_params
    ):
        super(NeuroSpex, self).__init__()
        self.eeg_encoder = EEGEncoder( **eeg_encoder_params ) 
        self.speech_encoder = SpeechEncoder( **speech_encoder_params )
        self.speech_decoder = SpeechDecoder( **speech_decoder_params )
        self.speaker_extractor = SpeakerExtractor( **spk_ext_params )

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
        spk_out = mask * speech_emb

        # Decode
        out = self.speech_decoder(spk_out)
        return out[..., :-10]