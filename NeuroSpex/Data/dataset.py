# Creating datasets

import os
import torch
import numpy as np
import pandas as pd
import librosa
import pickle
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, root, split='train', audio_sr=8000, eeg_sr=128):
        """
        Args:
            root (string): Directory with all the data files.
            split (string): One of 'train', 'val', 'test' to specify the dataset split.
        """
        self.audio_dir = os.path.join(root, 'stimuli')
        self.eeg_dir = os.path.join(root, 'eeg')
        self.manifest = pd.read_csv(os.path.join(root, f'{split}_manifest.csv'))
        self.audio_sr = audio_sr
        self.eeg_sr = eeg_sr

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):

        # Get metadata from manifest
        row = self.manifest.iloc[idx]
        stimuli_f = row['stimuli']
        eeg_f = f"{row['subject']}_{row['trial']}.pkl"
        pair_f = row['stimuli_pair']        
        seg_st = row['start']
        seg_en = row['end']

        # Load eeg
        eeg_path = os.path.join(self.eeg_dir, eeg_f)
        with open(eeg_path, 'rb') as f:
            eeg_data = torch.tensor(pickle.load(f))
        
        assert eeg_data.shape[-1] == 64, "EEG data should have 64 channels"

        # Load audio
        stimuli_path = os.path.join(self.audio_dir, stimuli_f)
        stimuli, _ = librosa.load(stimuli_path, sr=self.audio_sr)

        mixture_wavs = []
        for wav in pair_f.split():
            wav = os.path.join(self.audio_dir, wav)
            wav_i, _ = librosa.load(wav, sr=self.audio_sr)
            mixture_wavs.append(wav_i)

        # Mix the audio signals with a gain of 0dB
        mixture = mixture_wavs[0] + mixture_wavs[1]

        # Normalize the mixture to prevent clipping
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            mixture = mixture / max_val

        # Segment the audio and eeg
        stimuli = stimuli[int(seg_st*self.audio_sr):int(seg_en*self.audio_sr)]
        mixture = mixture[int(seg_st*self.audio_sr):int(seg_en*self.audio_sr)]
        eeg = eeg_data[int(seg_st*self.eeg_sr):int(seg_en*self.eeg_sr), :]

        return torch.tensor(mixture).unsqueeze(0), torch.tensor(stimuli).unsqueeze(0), eeg.permute(1, 0)
