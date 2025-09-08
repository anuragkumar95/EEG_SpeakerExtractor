import pickle
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# This scripts creates the train, val and test manifests for the KUL-Mix dataset
# Each segment is 4 sec long with 1 sec hop for train set
# Change the directories as needed
# Current implementation downsamples audio to 8kHz. Change as needed

np.random.seed(42)

NUM_TRIALS = 8
NUM_SUBJ = 16
EEG_PATH = "/fs/scratch/PAS2301/kumar1109/KUL-Mix/eeg"
WAV_PATH = "/fs/scratch/PAS2301/kumar1109/KUL-Mix/stimuli"
TRIALS_CSV = "/fs/scratch/PAS2301/kumar1109/KUL-Mix/trials.csv"
SAVE_PATH = "/fs/scratch/PAS2301/kumar1109/KUL-Mix"

EEG_SR=128
AUD_SR=8000

WIN_DUR=4
HOP_DUR=1

test_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
val_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
train_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
trials_csv = pd.read_csv(TRIALS_CSV)

# Since segments are 4sec long with 1 sec hop, select 357 random start points
print(f"Generating test manifest")
test_t_id = np.random.choice(range(1, NUM_TRIALS+1), NUM_SUBJ, replace=True)
for s_id, t_id in tqdm(enumerate(test_t_id)):
    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{s_id+1}") & (trials_csv['trial']==t_id)]
    track_id = str(list(trial_row['attended_track'])[0])
    stimulies = list(trial_row['stimuli'])[0].split()
    for stimuli in stimulies:
        s_track = stimuli.split('_')[1]
        if track_id in s_track:
            stimuli_f = stimuli
            break
    assert stimuli_f is not None, f"Stimuli:{stimuli_f}. Matching stimuli not found. Found stimulies:{stimulies}" 
    
    # Compare both stimuli audio lens to make sure eeg segments are made during the intervals where both tracks had audio     
    min_len = float('inf')
    for f_wav in stimulies:
        s_wav_f = f"{WAV_PATH}/{f_wav}"
        s_wav, sr = librosa.load(s_wav_f, sr=AUD_SR)
        min_len = min(min_len, s_wav.shape[-1])
    min_wav_dur = min_len / AUD_SR
    
    eeg_f = f"S{s_id+1}_{t_id}.pkl"
    with open(f"{EEG_PATH}/{eeg_f}", 'rb') as f:
        eeg = np.asarray(pickle.load(f))
        
    eeg_dur = eeg.shape[0] / EEG_SR
    if eeg_dur > min_wav_dur:
        eeg_end_idx = int(min_wav_dur * EEG_SR)
        eeg = eeg[:eeg_end_idx, :]
   
    win_len = WIN_DUR*EEG_SR
    t_starts = np.random.choice(eeg.shape[0]-(win_len), 357, replace=False)
    for t_st in t_starts:
        test_dict['subject'].append(f"S{s_id+1}")
        test_dict['trial'].append(t_id)
        test_dict['stimuli'].append(stimuli_f)
        test_dict['stimuli_pair'].append(" ".join(stimulies))
        test_dict['start'].append("{0:.4f}".format(t_st/EEG_SR))
        test_dict['end'].append("{0:.4f}".format((t_st/EEG_SR) + WIN_DUR))
        test_dict['dur'].append(WIN_DUR)  

# Store the ids in the format {subj}_{trial}
print(f"Generating validation manifest")
remaining_ids = [f"{i+1}_{k+1}" for i in range(NUM_SUBJ) for k in range(NUM_TRIALS) if k+1 != test_t_id[i]]
val_t_id = np.random.choice(remaining_ids, 4)
for t_id in tqdm(val_t_id):
    subj, trial = t_id.split('_')

    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{subj}") & (trials_csv['trial']==int(trial))]
    track_id = str(list(trial_row['attended_track'])[0])
    stimulies = list(trial_row['stimuli'])[0].split()
    for stimuli in stimulies:
        s_track = stimuli.split('_')[1]
        if track_id in s_track:
            stimuli_f = stimuli
            break
    assert stimuli_f is not None, f"Stimuli:{stimuli_f}. Matching stimuli not found. Found stimulies:{stimulies}" 
    
    # Compare both stimuli audio lens to make sure eeg segments are made during the intervals where both tracks had audio     
    min_len = float('inf')
    for f_wav in stimulies:
        s_wav_f = f"{WAV_PATH}/{f_wav}"
        s_wav, sr = librosa.load(s_wav_f, sr=AUD_SR)
        min_len = min(min_len, s_wav.shape[-1])
    min_wav_dur = min_len / AUD_SR

    eeg_f = f"S{subj}_{trial}.pkl"
    with open(f"{EEG_PATH}/{eeg_f}", 'rb') as f:
        eeg = np.asarray(pickle.load(f))

    eeg_dur = eeg.shape[0] / EEG_SR
    if eeg_dur > min_wav_dur:
        eeg_end_idx = int(min_wav_dur * EEG_SR)
        eeg = eeg[:eeg_end_idx, :]
        
    win_len = WIN_DUR*EEG_SR
    t_starts = np.random.choice(eeg.shape[0]-(win_len), 357, replace=False)
    for t_st in t_starts:
        val_dict['subject'].append(f"S{subj}")
        val_dict['trial'].append(int(trial))
        val_dict['stimuli'].append(stimuli_f)
        val_dict['stimuli_pair'].append(" ".join(stimulies))
        val_dict['start'].append("{0:.4f}".format(t_st/EEG_SR))
        val_dict['end'].append("{0:.4f}".format((t_st/EEG_SR) + WIN_DUR))
        val_dict['dur'].append(WIN_DUR) 

# Create the train dataset
print(f"Generating train manifest")
train_t_id = [f"{i+1}_{k+1}" for i in range(NUM_SUBJ) for k in range(NUM_TRIALS) if k+1 != test_t_id[i] and f"{i+1}_{k+1}" not in val_t_id]
for t_id in tqdm(train_t_id):
    subj, trial = t_id.split('_')

    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{subj}") & (trials_csv['trial']==int(trial))]
    track_id = str(list(trial_row['attended_track'])[0])
    stimulies = list(trial_row['stimuli'])[0].split()
    for stimuli in stimulies:
        s_track = stimuli.split('_')[1]
        if track_id in s_track:
            stimuli_f = stimuli
            break
    assert stimuli_f is not None, f"Stimuli:{stimuli_f}. Matching stimuli not found. Found stimulies:{stimulies}" 
    
    # Compare both stimuli audio lens to make sure eeg segments are made during the intervals where both tracks had audio     
    min_len = float('inf')
    for f_wav in stimulies:
        s_wav_f = f"{WAV_PATH}/{f_wav}"
        s_wav, sr = librosa.load(s_wav_f, sr=AUD_SR)
        min_len = min(min_len, s_wav.shape[-1])
    min_wav_dur = min_len / AUD_SR

    #Process EEG
    eeg_f = f"S{subj}_{trial}.pkl"
    with open(f"{EEG_PATH}/{eeg_f}", 'rb') as f:
        eeg = np.asarray(pickle.load(f))
        
    eeg_dur = eeg.shape[0] / EEG_SR
    if eeg_dur > min_wav_dur:
        eeg_end_idx = int(min_wav_dur * EEG_SR)
        eeg = eeg[:eeg_end_idx, :]
    
    win_len = WIN_DUR * EEG_SR
    hop_len = HOP_DUR * EEG_SR
    for t_st in range(0, eeg.shape[0], hop_len):
        t_en = t_st + win_len
        train_dict['subject'].append(f"S{subj}")
        train_dict['trial'].append(int(trial))
        train_dict['stimuli'].append(stimuli_f)
        train_dict['stimuli_pair'].append(" ".join(stimulies))
        train_dict['start'].append("{0:.4f}".format(t_st/EEG_SR))
        train_dict['end'].append("{0:.4f}".format(t_en/EEG_SR))
        train_dict['dur'].append(WIN_DUR)

print(f"TEST LEN:{len(test_dict['subject'])}")
print(f"VAL LEN:{len(val_dict['subject'])}")
print(f"TRAIN LEN:{len(train_dict['subject'])}")

test_df = pd.DataFrame.from_dict(test_dict)
val_df = pd.DataFrame.from_dict(val_dict)
train_df = pd.DataFrame.from_dict(train_dict)
test_df.to_csv(f"{SAVE_PATH}/test_manifest.csv", index=False)
val_df.to_csv(f"{SAVE_PATH}/val_manifest.csv", index=False)
train_df.to_csv(f"{SAVE_PATH}/train_manifest.csv", index=False)