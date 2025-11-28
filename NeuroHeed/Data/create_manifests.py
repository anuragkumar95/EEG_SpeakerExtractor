"""
Author: Anurag Kumar
Created on: 2025-09-07
"""

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
MIX_CSV = "/fs/scratch/PAS2301/kumar1109/KUL-Mix/mixture_data_list_2mix.csv"

EEG_SR=128
AUD_SR=8000

WIN_DUR=4
HOP_DUR=1
MAX_DUR=360  #seconds. Keep only first 6 mins of eeg and audio data

test_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
val_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
train_dict = {'subject':[],'trial':[],'stimuli':[],'stimuli_pair':[],'start':[],'end':[],'dur':[]}
trials_csv = pd.read_csv(TRIALS_CSV)
mix_csv = pd.read_csv(MIX_CSV, header=None)

# Since segments are 4sec long with 1 sec hop, select 357 random start points
print(f"Generating test manifest")
counter = {i:0 for i in range(1, NUM_TRIALS+1)}
test_t_id = []
while(len(test_t_id) < NUM_SUBJ):
    r_trial = int(np.random.choice(range(1, NUM_TRIALS+1)))
    if counter[r_trial] < 2:
        counter[r_trial] += 1
        test_t_id.append(r_trial) 

s_id = 0
for t_id in tqdm(test_t_id):
    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{s_id+1}") & (trials_csv['trial']==t_id)]
    mix_row = mix_csv[(mix_csv[1]==s_id+1) & (mix_csv[2]==t_id)]
    stimuli_f = mix_row[3].values[0]
    stimulies = f"{mix_row[3].values[0]} {mix_row[6].values[0]}"

    for t_st in range(0, MAX_DUR-WIN_DUR+1, HOP_DUR):
        t_en = min(t_st + WIN_DUR, MAX_DUR)
        dur = t_en - t_st
        test_dict['subject'].append(f"S{s_id+1}")
        test_dict['trial'].append(t_id)
        test_dict['stimuli'].append(stimuli_f)
        test_dict['stimuli_pair'].append(stimulies)
        test_dict['start'].append(t_st)
        test_dict['end'].append(t_en)
        test_dict['dur'].append(dur)  
    
    s_id += 1

# Store the ids in the format {subj}_{trial}
print(f"Generating validation manifest")
remaining_ids = [f"{i+1}_{k+1}" for i in range(NUM_SUBJ) for k in range(NUM_TRIALS) if k+1 != test_t_id[i]]
val_t_id = np.random.choice(remaining_ids, 4)
for t_id in tqdm(val_t_id):
    subj, trial = [int(i) for i in t_id.split('_')]

    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{subj}") & (trials_csv['trial']==trial)]
    mix_row = mix_csv[(mix_csv[1]==subj) & (mix_csv[2]==trial)]

    stimuli_f = mix_row[3].values[0]
    stimulies = f"{mix_row[3].values[0]} {mix_row[6].values[0]}"
        
    for t_st in range(0, MAX_DUR-WIN_DUR+1, HOP_DUR):
        t_en = min(t_st + WIN_DUR, MAX_DUR)
        dur = t_en - t_st
        val_dict['subject'].append(f"S{subj}")
        val_dict['trial'].append(int(trial))
        val_dict['stimuli'].append(stimuli_f)
        val_dict['stimuli_pair'].append(stimulies)
        val_dict['start'].append(t_st)
        val_dict['end'].append(t_en)
        val_dict['dur'].append(dur) 

# Create the train dataset
print(f"Generating train manifest")
train_t_id = [f"{i+1}_{k+1}" for i in range(NUM_SUBJ) for k in range(NUM_TRIALS) if k+1 != test_t_id[i] and f"{i+1}_{k+1}" not in val_t_id]
for t_id in tqdm(train_t_id):
    subj, trial = [int(i) for i in t_id.split('_')]

    # Get the right stimuli off the pair
    stimuli_f = None
    trial_row = trials_csv[(trials_csv['subject']==f"S{subj}") & (trials_csv['trial']==trial)]
    mix_row = mix_csv[(mix_csv[1]==subj) & (mix_csv[2]==trial)]
    stimuli_f = mix_row[3].values[0]
    stimulies = f"{mix_row[3].values[0]} {mix_row[6].values[0]}"

    for t_st in range(0, MAX_DUR-WIN_DUR+1, HOP_DUR):
        t_en = min(t_st + WIN_DUR, MAX_DUR)
        dur = t_en - t_st
        train_dict['subject'].append(f"S{subj}")
        train_dict['trial'].append(int(trial))
        train_dict['stimuli'].append(stimuli_f)
        train_dict['stimuli_pair'].append(stimulies)
        train_dict['start'].append(t_st)
        train_dict['end'].append(t_en)
        train_dict['dur'].append(dur)

print(f"Test size: {len(test_dict['subject'])}, Val size: {len(val_dict['subject'])}, Train size: {len(train_dict['subject'])}")

test_df = pd.DataFrame.from_dict(test_dict)
val_df = pd.DataFrame.from_dict(val_dict)
train_df = pd.DataFrame.from_dict(train_dict)
test_df.to_csv(f"{SAVE_PATH}/test_manifest.csv", index=False)
val_df.to_csv(f"{SAVE_PATH}/val_manifest.csv", index=False)
train_df.to_csv(f"{SAVE_PATH}/train_manifest.csv", index=False)