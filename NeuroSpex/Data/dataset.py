# Creating datasets
import soundfile as sf
import os
import math
import torch
import numpy as np
import pandas as pd
import librosa
import pickle
from torch.utils.data import Dataset

class NeurHeedEEG_Dataset(Dataset):
    def __init__(self, root, batch_size=16, partition='train', max_length=10, audio_sr=8000, ref_sr=128):
        self.minibatch =[]
        #self.args = args
        self.partition = partition
        self.max_length = max_length
        self.audio_sr=audio_sr
        self.ref_sr=ref_sr
        #self.speaker_no=args.speaker_no
        self.batch_size=batch_size

        #self.mix_lst_path = args.mix_lst_path
        self.mix_lst_path = f"{root}/mixture_data_list_2mix.csv"
        #self.audio_direc = args.audio_direc
        self.audio_direc = f"{root}/audio_8k/"
        #self.eeg_direc = args.reference_direc
        self.eeg_direc = f"{root}/eeg/"
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

        self.eeg_dict={}
        for subject in range(1,17):
            for trial in range(1,9):
                eeg_path = f'{self.eeg_direc}S{subject}Tra{trial}.npy'
                #eeg_path = f"{self.eeg_direc}S{subject}_{trial}.npy"
                eeg_data = np.load(eeg_path)
                self.eeg_dict[(subject,trial)] = eeg_data



    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_eegs=[]
        
        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1].split(',')[-1])      # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second*self.ref_sr)
        min_length_audio = math.floor(min_length_second*self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length*self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length*self.audio_sr)

        for line_cache in batch_lst:
            line=line_cache.split(',')

            # load target eeg
            subject, trial = line[1], line[2]
            eeg_data = self.eeg_dict[(int(subject),int(trial))]
            eeg_start = int(float(line[4])*self.ref_sr)
            eeg_end = eeg_start + min_length_eeg
            eeg_tgt = eeg_data[eeg_start:eeg_end,:]

            # load tgt audio
            tgt_audio_path = self.audio_direc + line[3]
            start = float(line[4]) * self.audio_sr
            end = start + min_length_audio
            a_tgt, _ = sf.read(tgt_audio_path, start=int(start), stop=int(end), dtype='float32')

            # load int eeg
            int_audio_path = self.audio_direc + line[6]
            start = float(line[7]) * self.audio_sr
            end = start + min_length_audio
            a_int, _ = sf.read(int_audio_path, start=int(start), stop=int(end), dtype='float32')

            # training snr augmentation
            if float(line[8]) != 0:
                target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
                intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
                a_int *= np.sqrt(target_power/intef_power)
                snr_1 = (10**(float(line[8])/20))

                max_snr = max(1, snr_1)
                a_tgt /= max_snr
                a_int /= max_snr
                a_int = a_int * snr_1

            a_mix = a_tgt + a_int

            # audio normalization
            max_val = np.max(np.abs(a_mix))
            if max_val > 1:
                a_mix /= max_val
                a_tgt /= max_val

            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_eegs.append(eeg_tgt)

        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), np.asarray(tgt_eegs, dtype=np.float32)


    def __len__(self):
        return len(self.minibatch)


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
        eeg_f = f"{row['subject']}_{row['trial']}.npy"
        pair_f = row['stimuli_pair']        
        seg_st = row['start']
        seg_en = row['end']

        # Load eeg
        eeg_path = os.path.join(self.eeg_dir, eeg_f)
        eeg_data = np.load(eeg_path)
        assert eeg_data.shape[-1] == 64, "EEG data should have 64 channels"
        eeg = eeg_data[int(seg_st*self.eeg_sr):int(seg_en*self.eeg_sr), :]

        # Load audio
        stimuli_path = os.path.join(self.audio_dir, stimuli_f)
        stimuli, _ = librosa.load(stimuli_path, sr=self.audio_sr)
        stimuli = stimuli[int(seg_st*self.audio_sr):int(seg_en*self.audio_sr)]

        mixture_wavs = []
        for wav in pair_f.split():
            wav = os.path.join(self.audio_dir, wav)
            wav_i, _ = librosa.load(wav, sr=self.audio_sr)
            mixture_wavs.append(wav_i[int(seg_st*self.audio_sr):int(seg_en*self.audio_sr)])
        assert (mixture_wavs[0] - stimuli).sum() == 0, "Stimuli not the same as attnding mixture audio"
        assert mixture_wavs[0].shape == mixture_wavs[1].shape, "Mixture segments have different shapes"

        # Mix the audio signals with a gain of 0dB
        mixture = mixture_wavs[0] + mixture_wavs[1]

        # Normalize the mixture to prevent clipping
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            mixture = mixture / max_val
            stimuli = stimuli / max_val

        return torch.tensor(mixture).unsqueeze(0), torch.tensor(stimuli).unsqueeze(0), torch.FloatTensor(eeg).permute(1, 0)
